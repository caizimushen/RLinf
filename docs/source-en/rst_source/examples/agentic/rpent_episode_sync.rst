RPent Episode PPO and GRPO
==========================

Train a Qwen3.5-4B planner while remote LIBERO-PRO environments execute its tool calls.
Use the same full multimodal SFT checkpoint and the same RPent workflow for rollout,
training and evaluation. This recipe uses the ``episode`` workflow with explicit
perception and ordered batches of at most three actions.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models

      Qwen3.5-4B planner; frozen vision encoder and projector

   .. grid-item-card:: Algorithms

      Synchronous GRPO, or PPO with an independent critic

   .. grid-item-card:: Tasks

      LIBERO-PRO Goal, Spatial and Object

   .. grid-item-card:: Hardware

      Four 8-H100 model hosts and one 8-A100 environment host

Tasks
~~~~~

The supplied GRPO configuration runs 150 updates. Each update selects whole groups.

.. list-table::
   :header-rows: 1

   * - Item
     - Setting
   * - Training pool
     - Native states of 30 tasks; duplicate states within a task are removed, preserving original state IDs
   * - GRPO batch
     - 32 groups × 8 episodes = 256 episodes; all eight members share a task and native initial state
   * - Oversampling
     - Start 40 groups × 8 episodes; select the first 32 complete groups, cancel the other 64 episodes, confirm generation has stopped and release all 320 environments before learning
   * - Reward
     - Native success 1, normal failure 0; retain all-zero and all-one groups
   * - Regression panel
     - 30 tasks × seeds 0 and 1 = 60 training-pool episodes

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

The workflow preserves the SFT planner's prompts, tool schemas and history policy.

.. list-table::
   :header-rows: 1

   * - Item
     - Contract
   * - Observation
     - Explicit camera observations at 1024 pixels; a turn before the first observation can contain no image
   * - Action
     - ``execute_actions`` with at most three ordered actions; ``view_env``, ``segment``, ``back_project`` and the ``finish`` output tool
   * - Reward
     - Environment success; a planner's ``finish`` response does not establish success
   * - Prompt
     - Thinking enabled without an effort tier; preserve reasoning and images, compact at 70% of context or 80 messages and retain 10 messages
   * - Limits
     - 262144 context tokens, up to 16384 output tokens per request, 50 requests, 10000 environment steps, 7200 seconds per planner and 14400 seconds per collector

How GRPO Works
--------------

The learner computes advantages from each group's mean and standard deviation.
It keeps every selected group, including groups whose rewards are identical.
There is no critic or second batch-wide advantage normalization.
The configuration uses group-, agent- and turn-level loss scaling, token-mean
aggregation, clip bounds 0.2/0.28, learning rate 1e-6 with zero warmup, zero KL
and entropy coefficients, gradient clipping at 1 and one pass per batch.
Sampled reasoning and tool-call tokens contribute to learning; prompt, tool
feedback, image and padding tokens have mask zero.

Selecting faster complete groups can bias which tasks reach training. Compare
selected and cancelled groups using the sampling manifests. Different training
batches are not a controlled evaluation of policy improvement.

Installation
------------

Prepare Linux x86-64, Git, ``uv``, a C++ compiler, Rust, ``patch`` and a
CUDA 13 development toolkit on every model host. Use H100 or newer GPUs for the
FA4 training path and a driver compatible with the CUDA 13 wheels. The environment
host needs EGL/OpenGL and the separate CUDA 12.8 simulator environment below.

An existing RLinf ``reason`` image can supply system tools, but its default
Python environment does not contain this TokenSpace stack. Build a separate
environment inside a CUDA 13 development image or on a prepared host. Container
build options are in ``docker/README.md``; the reproducible path here uses the
installer rather than relying on an unversioned pre-existing image.

Choose a shared root and clone the ``rpent-ppo`` branch of both repositories. Record immutable revisions
when deploying a run.

.. code-block:: bash

   export STACK_ROOT="$HOME/rpent-stack"
   mkdir -p "$STACK_ROOT/repos"
   git clone --branch rpent-ppo https://github.com/zhexuanxu/RLinf.git "$STACK_ROOT/repos/RLinf"
   git clone --branch rpent-ppo https://github.com/zhexuanxu/RPent.git "$STACK_ROOT/repos/RPent"
   export RLINF_ROOT="$STACK_ROOT/repos/RLinf"
   export RPENT_ROOT="$STACK_ROOT/repos/RPent"
   git -C "$RLINF_ROOT" rev-parse HEAD
   git -C "$RPENT_ROOT" rev-parse HEAD

On each model host, install the explicit TokenSpace profile into a new environment:

.. code-block:: bash

   cd "$RLINF_ROOT"
   UV_TORCH_BACKEND=cu130 UNINSTALL_FA4=0 bash requirements/install.sh agentic \
     --engine sglang --sglang token-space --torch 2.14.1 \
     --transformers 5.19.0 --python 3.12.12 \
     --venv "$STACK_ROOT/envs/model" --install-rlinf
   source "$STACK_ROOT/envs/model/bin/activate"
   uv pip freeze > "$STACK_ROOT/model-packages.txt"

The profile pins SGLang commit ``18a44326f6de5134fe921927d760d81f8ee48fc9``,
Torch 2.14.1 and Transformers 5.19.0. It installs Transformer Engine 2.17,
Megatron-Bridge 0.5 and the Megatron 0.18 source, including the CPU-offload
checkpoint fix. With an explicit ``MEGATRON_PATH``, use commit
``dc0ee41ada177d4c74ba46bb501bba5967264815`` and apply
the following patch yourself:

.. code-block:: text

   requirements/agentic/megatron-core-r0.18.0-cpu-offload.patch

The installer also applies the FA4 window-sentinel adapter from
`Transformer Engine PR 3540 <https://github.com/NVIDIA/TransformerEngine/pull/3540>`__.
It changes the FA4 Python boundary, keeping the native attention kernels.
Keep FA2 and FA4 in separate environments. Preserve the complete environment,
Python sources, dependency source trees and JIT headers on each node; copying
only a subset of site-packages is insufficient for source-inspecting compilers.
Never replace or prune an environment used by an active job.

On each model host, install a separate CPU planner environment. It runs RPent's
model client and tools over RPC, using the local checkpoint processor to count
prompt tokens. It does not load the planner weights or simulator:

.. code-block:: bash

   cd "$RPENT_ROOT"
   uv venv --python 3.12 "$STACK_ROOT/envs/planner"
   uv pip install --python "$STACK_ROOT/envs/planner/bin/python" \
     --prerelease=allow --torch-backend cpu -r requirements/episode-planner.txt

Set ``tools.rpent.python`` to ``$STACK_ROOT/envs/planner/bin/python``.
Reactivate the model environment before starting Ray.

On the environment host, install the simulator and SAM3 independently:

.. code-block:: bash

   cd "$RPENT_ROOT"
   export RLINF_ROOT
   uv venv --python 3.12 "$STACK_ROOT/envs/environment"
   source "$STACK_ROOT/envs/environment/bin/activate"
   uv pip install --prerelease=allow --torch-backend cu128 \
     -r requirements/episode-ppo-env.txt
   export PYTHONPATH="$RPENT_ROOT:$RLINF_ROOT${PYTHONPATH:+:$PYTHONPATH}"
   export LIBERO_TYPE=pro MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

``episode-ppo-env.txt`` is the shared episode environment for PPO and GRPO.
It supplies LIBERO/LIBERO-PRO, SAM3, MuJoCo, robosuite and the RPC dependencies.
Follow RPent's ``tools/rl/INSTALL.md`` to download LIBERO-PRO assets and create
``LIBERO_CONFIG_PATH/config.yaml``. Keep the simulator's Transformers 4.57
installation separate from the model host's Transformers 5.19.

Download the Model
------------------

Prepare the following assets before launching services. Python installation
does not download model weights or scene assets.

.. list-table::
   :header-rows: 1

   * - Asset
     - Used by
     - Preparation
   * - Full Qwen3.5-4B SFT directory
     - RLinf rollout and learner
     - Weights, config, tokenizer, chat template and processor at the same absolute path on all model hosts
   * - SAM3 checkpoint
     - Environment host's SAM3 services
     - Request access to ``facebook/sam3``, then download ``sam3.pt``
   * - LIBERO-PRO assets
     - Environment workers
     - Scene assets plus the package's BDDL files and native initial states

For an SFT comparison, point ``MODEL_PATH`` to the original full multimodal
SFT directory, including its processor and tokenizer files:

.. code-block:: bash

   export MODEL_PATH=/absolute/path/to/full-sft-checkpoint

For a base-model run, prepare a separate model directory instead:

.. code-block:: bash

   export MODEL_PATH="$STACK_ROOT/models/Qwen3.5-4B"
   hf download Qwen/Qwen3.5-4B --local-dir "$MODEL_PATH"

Download the SAM3 checkpoint on the environment host:

.. code-block:: bash

   export SAM3_CHECKPOINT_PATH="$STACK_ROOT/models/sam3/sam3.pt"
   hf download facebook/sam3 sam3.pt --local-dir "$(dirname "$SAM3_CHECKPOINT_PATH")"

The direct-control ``episode`` workflow needs no Pi0.5 weights. Pi0.5 belongs
to the separate :doc:`original planner recipe <rpent>`; changing to that recipe
changes the model's action tools. For a checkpoint trained with the episode workflow, keep ``workflow=episode``.

Run It
------

Start SAM3 and the Environment Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run two SAM3 services per environment GPU, each with a distinct local port.
For example, the first GPU's two instances can use ports 8114 and 8115:

.. code-block:: bash

   python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8114
   python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8115

Run the commands in separate terminals or under your service supervisor.
Repeat for GPUs 1–7 with unique ports, then pass all 16 endpoints to
``rpent.robots.components.sam3_pool --backends``; its complete service example
is in RPent's ``tools/rl/WORKFLOWS.md``. Set ``SAM_POOL`` to the pool URL
and start the environment backend:

.. code-block:: bash

   python -m robots.rl.remote_manager \
     --output-dir "$STACK_ROOT/environment-runs" \
     --gpu-slots 0:40,1:40,2:40,3:40,4:40,5:40,6:40,7:40 \
     --max-initializing-per-gpu 2 --sam3-endpoint "$SAM_POOL" \
     --host 127.0.0.1 --port 28490

The backend retains 320 slots. Completed episodes release slots throughout
sampling, so an instantaneous ready count can be lower than 320.
During learning all environment slots should be released; SAM3 remains resident.
Capacity depends on GPU and host memory as well as slot counts.

Expose this loopback backend on every model host through an authenticated private
route. For an SSH reverse forward, run this on the environment server for each
model host, with pinned host keys:

.. code-block:: bash

   ssh -NT -o ExitOnForwardFailure=yes \
     -R 127.0.0.1:28490:127.0.0.1:28490 model-host-0

Supervise the forwards on the environment server. The four model hosts join
one private Ray cluster with distinct ``RLINF_NODE_RANK`` values, as described
in :doc:`/rst_source/guides/multi_node`. The environment host stays outside
Ray. Activate the matching model environment and export its paths before
starting Ray; run the training entry script only on the Ray head.

Prepare Tasks and Local Paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate task files on the environment host, where LIBERO-PRO is installed:

.. code-block:: bash

   export TASK_DIR="$STACK_ROOT/tasks"
   python "$RLINF_ROOT/examples/agent/rpent/prepare_tasks.py" \
     --workflow episode --libero-type pro --seeds all --split train \
     --libero-config-path "$LIBERO_CONFIG_PATH" --output-dir "$TASK_DIR"
   python "$RLINF_ROOT/examples/agent/rpent/prepare_tasks.py" \
     --workflow episode --libero-type pro --seeds 0,1 --split eval \
     --libero-config-path "$LIBERO_CONFIG_PATH" --output-dir "$TASK_DIR"

Copy the task directory and manifests to the model hosts' shared filesystem.
The 60-episode panel measures training-pool regression, not generalization.

Create an external deployment file from the supplied template:

.. code-block:: bash

   export CONFIG_DIR="$STACK_ROOT/config"
   mkdir -p "$CONFIG_DIR/deployment"
   cp "$RLINF_ROOT/examples/agent/rpent/config/deployment/episode.yaml" \
     "$CONFIG_DIR/deployment/local.yaml"

Fill every ``???`` in ``local.yaml``. Use the generated training and evaluation
JSONL files for ``data.train_data_paths`` and ``data.val_data_paths``.
Write dataset paths as lists under ``data``:

.. code-block:: yaml

   data:
     train_data_paths: ["/data/tasks/train.jsonl"]
     val_data_paths: ["/data/tasks/eval.jsonl"]

Point each of these model paths to the same full SFT directory:

.. code-block:: text

   rollout.model.model_path
   actor.tokenizer.tokenizer_model
   critic.tokenizer.tokenizer_model
   actor.megatron.ckpt_convertor.hf_model_path
   critic.megatron.ckpt_convertor.hf_model_path

Set ``tools.rpent.python`` to the planner interpreter and ``repo_path``
to the matching RPent checkout. The episode output and preprocessing cache
directories must have the same absolute paths on all model hosts. Keep image
files while trajectories or checkpoints reference them.

Launch Training
~~~~~~~~~~~~~~~

Start GRPO from the Ray head:

.. code-block:: bash

   cd "$RLINF_ROOT"
   source "$STACK_ROOT/envs/model/bin/activate"
   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_sync_grpo \
     --config-dir "$CONFIG_DIR" +deployment@_global_=local

Actor placement is TP2/CP1/DP16; rollout uses 32 TP1 replicas.
The configuration keeps ``NCCL_MAX_CTAS=8`` for actor/critic/rollout and
SGLang's memory fraction at 0.55. This fraction is not a total memory cap for
a GPU shared with the learner. Keep placement and memory settings fixed when comparing runs.

To use PPO with the same preprocessing environment, select the PPO recipe
and the shared TokenSpace configuration:

.. code-block:: bash

   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_sync_oversample320 \
     +preprocessing@_global_=token_space \
     --config-dir "$CONFIG_DIR" +deployment@_global_=local

PPO uses 320 independent candidates and the first 256 complete episodes,
an independent critic and chronological episode GAE. GRPO's group normalization
does not apply to that PPO recipe.

Inspect the first complete optimizer update, weight synchronization and the next
online episode before leaving a new installation unattended. A finished cohort
or a single TensorBoard tag does not establish a completed update. GPU numerical parity and compatibility with other driver or kernel versions
remain unverified for a new installation.

Resume, Export and Evaluate
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``runner.resume_dir`` to resume a complete checkpoint from the same
configuration. Check actor/critic model and optimizer shards as applicable,
scheduler/RNG state, and the atomically published data state. The dataset cursor
is local to its epoch; it need not equal the global update number. Preserve
partial raw episodes, and do not treat updates beyond the last complete
checkpoint as recoverable. See :doc:`/rst_source/guides/resume`.

After update 150, export the trained language policy into the original full
multimodal model:

.. code-block:: bash

   python examples/agent/rpent/checkpoint.py export \
     --checkpoint "$CHECKPOINT_150" --vlm "$MODEL_PATH" --output "$EXPORT_PATH"

Evaluate the export on the prepared panel with the matching model environment.
Use the TokenSpace rollout settings from the training profile:

.. code-block:: bash

   bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b_episode_token_space \
     --config-dir "$CONFIG_DIR" +deployment@_global_=local \
     rollout.model.model_path="$EXPORT_PATH"

Use an output directory separate from training. Keep the original SFT directory
and every checkpoint. Normal failures and output/context/turn/time limits have
zero reward; only classified infrastructure failures have the configured bounded
retry. A second exhausted infrastructure attempt must stop admission for review.

Visualization and Results
-------------------------

TensorBoard reports optimizer loss, gradient norm, learning rate, sampled
success, TIS diagnostics and update timings. The sampling manifests identify
the selected groups, cancelled members and cleanup receipts; compute success
rates from selected episodes rather than directory counts. The episode files
contain request/outcome data, image references and sampled training turns.

Full HTTP request/response and compaction traces are disabled for episode
training by default. Set ``tools.rpent.record_api_trace=true`` only when
investigating a planner transport problem; direct RPent launches use
``--record-api-trace``. This does not change the prompts, compaction decisions,
token masks or saved training records.

Report task failures, infrastructure failures, selected/cancelled group
composition, token/turn costs and the exact source/dependency revisions
separately. A 60-episode training-pool regression is not a held-out benchmark.
