RPent Episode PPO Across Clusters
=================================

Train the Qwen3.5-4B planner with synchronous PPO while a separate GPU host
executes the LIBERO-PRO environments and SAM3. Each update uses complete
episodes from the Goal, Spatial and Object suites. The ``episode`` workflow
uses ordered batches of at most three actions and explicit perception.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models

      Qwen3.5-4B, full multimodal checkpoint

   .. grid-item-card:: Algorithms

      Synchronous PPO with an independent critic

   .. grid-item-card:: Tasks

      LIBERO-PRO Goal, Spatial and Object

   .. grid-item-card:: Hardware

      Two 8-GPU training hosts and one 8-GPU environment host

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Item
     - Setting
   * - Training pool
     - All native initial states of 30 tasks; duplicate states within each task are removed without renumbering their IDs
   * - PPO batch
     - 256 complete episodes, including normal task failures; one update pass
   * - Regression panel
     - The same 30 tasks at seeds 0 and 1, 60 episodes; these states overlap training
   * - Reward
     - Native environment success 1, normal task failure 0; one terminal reward per episode

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Item
     - Contract
   * - Tools
     - ``execute_actions``, ``view_env``, ``segment``, ``back_project`` and ``finish``
   * - Images
     - Explicit paired-camera observations at 1024 pixels; images and depth retain their recorded frame identity
   * - History
     - Keep reasoning and image history; compact at 70% of context or 80 messages, retaining 10 messages
   * - Limits
     - 80 model requests, 10,000 environment steps, 262,144 context tokens and 32,768 output tokens per request

The planner and image preprocessing run on the training hosts. Only environment
and SAM3 execution crosses the private RPC route. Training images are copied
and hash-checked into the shared episode directory on the training hosts.
Prompt tokens, tool results and images have loss mask zero; sampled reasoning and tool-call
tokens retain their original IDs and behavior logprobs.

Installation
------------

Use matching validated RPent revisions on both sides, and an RLinf checkout
containing this recipe. Set ``RLINF_ROOT`` and ``RPENT_ROOT`` to those checkouts.
Install the agentic Megatron/SGLang environment on both training hosts:

.. code-block:: bash

   cd "$RLINF_ROOT"
   UV_TORCH_BACKEND=cu128 UNINSTALL_FA4=1 bash requirements/install.sh agentic \
     --engine sglang --python 3.12.12 --torch 2.11.0 \
     --sglang 0.5.12.post1 --transformers 5.6.1 --install-rlinf
   source .venv/bin/activate
   uv pip install -e "$RPENT_ROOT"

Keep the training hosts on the same Python, Torch, SGLang, Transformers,
Megatron-Core and Bridge versions. Verify CUDA, Transformer Engine and
FlashAttention imports before loading weights. See :doc:`rpent_libero_rl_setup`
for model/environment installation prerequisites.

On the environment host, use the dedicated simulator/SAM3 requirements with
``RLINF_ROOT`` exported. They install the paired checkout without OpenPi:

.. code-block:: bash

   cd "$RPENT_ROOT"
   export RLINF_ROOT
   uv venv --python 3.12 .env-ppo
   source .env-ppo/bin/activate
   uv pip install --prerelease=allow --torch-backend cu128 \
     -r requirements/episode-ppo-env.txt
   export PYTHONPATH="$RPENT_ROOT:$RLINF_ROOT${PYTHONPATH:+:$PYTHONPATH}"

Install LIBERO-PRO assets and configure their locations as described in
RPent's ``tools/rl/INSTALL.md``. Export ``LIBERO_TYPE=pro``,
``LIBERO_CONFIG_PATH``, ``MUJOCO_GL=egl`` and ``PYOPENGL_PLATFORM=egl``.
Start two SAM3 instances per GPU and one SAM3 pool using
``rpent.robots.components.sam3_server`` and ``sam3_pool``; the service commands
are in RPent's ``tools/rl/WORKFLOWS.md``. Then start the remote backend:

.. code-block:: bash

   cd "$RPENT_ROOT"
   python -m robots.rl.remote_manager \
     --output-dir "$ENV_OUTPUT" \
     --gpu-slots 0:32,1:32,2:32,3:32,4:32,5:32,6:32,7:32 \
     --max-initializing-per-gpu 2 --sam3-endpoint "$SAM_POOL" \
     --host 127.0.0.1 --port 28490

``ENV_OUTPUT`` retains all backend attempts. ``SAM_POOL`` is the environment
host's local pool URL. Qualify the 256-environment/16-SAM3 layout with real
actions, perception bursts and at least five minutes of residency before
training. Slot counts alone do not establish available capacity.

Connect the environment listener to each training host with server-owned
reverse forwards. Run each command in a separate terminal on the environment
server, using SSH aliases you have configured and authenticated:

.. code-block:: bash

   ssh -NT -o ExitOnForwardFailure=yes \
     -R 127.0.0.1:28490:127.0.0.1:28490 <training-host-0>
   ssh -NT -o ExitOnForwardFailure=yes \
     -R 127.0.0.1:28490:127.0.0.1:28490 <training-host-1>

Supervise both forwards on the environment server. Keep private keys there
and pin host keys. The two training hosts share a private Ray/NCCL network
and a filesystem; follow :doc:`/rst_source/guides/multi_node` to start Ray
with a distinct ``RLINF_NODE_RANK`` on each host. The environment server
does not join that Ray cluster.

Download the Model
------------------

Download the original full multimodal checkpoint into the training hosts'
shared model directory:

.. code-block:: bash

   hf download Qwen/Qwen3.5-4B --local-dir "$MODEL_PATH"

``MODEL_PATH`` must resolve to the same weights, tokenizer and processor on
both training hosts. This recipe enables thinking; the 4B template does not
implement reasoning-effort tiers.

Run It
------

On the environment host, generate the native state pool and fixed regression
panel with its LIBERO configuration:

.. code-block:: bash

   python "$RLINF_ROOT/examples/agent/rpent/prepare_tasks.py" \
     --workflow episode --libero-type pro --seeds all --split train \
     --libero-config-path "$LIBERO_CONFIG_PATH" --output-dir "$TASK_DIR"
   python "$RLINF_ROOT/examples/agent/rpent/prepare_tasks.py" \
     --workflow episode --libero-type pro --seeds 0,1 --split eval \
     --libero-config-path "$LIBERO_CONFIG_PATH" --output-dir "$TASK_DIR"

Copy ``TASK_DIR`` to the training hosts' shared filesystem. Keep both manifests;
they record state IDs, state hashes and dataset hashes. The panel measures
regression on training-pool states, not held-out generalization.

On the Ray head, launch the synchronous recipe with explicit paths:

.. code-block:: bash

   cd "$RLINF_ROOT"
   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_sync_ppo \
     runner.output_dir="$RUN_DIR" \
     data.train_data_paths="[$TASK_DIR/train.jsonl]" \
     rollout.model.model_path="$MODEL_PATH" \
     tools.rpent.repo_path="$RPENT_ROOT" \
     tools.rpent.python="$RPENT_PYTHON" \
     tools.rpent.output_dir="$EPISODE_DIR" \
     tools.rpent.remote_toolkit_endpoint=http://127.0.0.1:28490

``RPENT_PYTHON`` runs the planner on each training host. ``EPISODE_DIR`` must
be shared by both learners and retained for checkpoint recovery. The recipe
places actor/critic at TP2, CP1, DP8 and rollout at 16 TP1 servers, with one
agent worker per rollout server and 16 episode slots per worker.

Start a separate correctness run with ``data.rollout_batch_size=8`` and
``runner.max_steps=2``. Save and restore at a complete update boundary using
``runner.resume_dir``. Then run three updates with 256 episodes per update,
starting from the original checkpoint with fresh
``RUN_DIR`` and ``EPISODE_DIR`` directories. Keep the
80-request and 32,768-output budgets in the small run too.

Evaluate the fixed 60-episode panel with the same workflow and sampling:

.. code-block:: bash

   bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b_episode_sync \
     runner.output_dir="$RUN_DIR" \
     data.val_data_paths="[$TASK_DIR/eval.jsonl]" \
     rollout.model.model_path="$MODEL_PATH" \
     tools.rpent.repo_path="$RPENT_ROOT" \
     tools.rpent.python="$RPENT_PYTHON" \
     tools.rpent.output_dir="$EPISODE_DIR" \
     tools.rpent.remote_toolkit_endpoint=http://127.0.0.1:28490

Set ``MODEL_PATH`` to the original checkpoint for the initial panel and to
the policy produced by ``checkpoint.py export`` for the final panel. Use
separate result and episode directories for each panel. Evaluation splits
all 60 episodes across the 16 workers without adding or dropping episodes.

Visualization and Results
-------------------------

The recipe uses temperature 1, top-p 1, disabled top-k, zero presence penalty
and no repetition penalty. Actor/critic learning rates are 1e-6/5e-6, with
gamma/lambda 1, clipping 0.2 and one data pass per update. These are starting
settings for validation, not an established optimum.

Inspect native success, complete-episode counts, fixed policy versions,
actor/critic losses and gradients, TIS mismatch, and model/perception/network
timing. Each episode preserves HTTP traces, sampled token audit and ordered
training images. Normal task failures and budget endings without native success,
including context exhaustion and output truncation, have zero terminal reward.
Infrastructure failures
retain their attempts and permit at most one retry; exhausted retries stop
the batch.

Before reporting training results, compare HF, Megatron and SGLang selected
token logprobs with identical IDs, image grids and masks. Check that language
and critic parameters change while frozen vision parameters remain unchanged,
and verify the installed policy version on every rollout server. Report the
same 60-episode panel at initialization and after update three. Compare timing
for matching task/seed pairs.
The patched SGLang backend keeps Qwen3.5 GDN output-norm scales in FP32.
Weight export also computes ``1 + actor_gamma`` in FP32 so that the rollout
retains the actor's effective scale. Keep the corresponding checkpoint
parameters in FP32 in the HF reference when checking numerical parity.
Three updates establish integration/stability evidence; they do not establish
an optimal PPO configuration or generalization improvement.


Run Synchronous Oversampling on 32 GPUs
---------------------------------------

Use four 8-GPU model hosts and an 8-GPU environment host for the 320-candidate preset. Actor and critic use TP2, CP1 and DP16; 32 TP1 rollout servers each have one agent worker with 10 episode slots. Qualify 40 environments and two SAM3 instances per environment GPU, including real actions, paired-camera perception, five minutes of residency and owned-process cleanup.

On an idle environment backend, use a separate output directory and raise its slot limits:

.. code-block:: bash

   python -m robots.rl.remote_manager \
     --output-dir "$ENV_OUTPUT" \
     --gpu-slots 0:40,1:40,2:40,3:40,4:40,5:40,6:40,7:40 \
     --max-initializing-per-gpu 2 --sam3-endpoint "$SAM_POOL" \
     --host 127.0.0.1 --port 28490

Keep the authenticated reverse forwards for all four model hosts and verify four private Ray/NCCL nodes with 32 GPUs. The paired RPent revision must support attempt-owned cancellation during environment startup. Cancelled attempts keep their backend artifacts and cleanup receipts.

Launch the preset with the same path variables as the base recipe:

.. code-block:: bash

   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_sync_oversample320 \
     runner.output_dir="$RUN_DIR" \
     data.train_data_paths="[$TASK_DIR/train.jsonl]" \
     rollout.model.model_path="$MODEL_PATH" \
     tools.rpent.repo_path="$RPENT_ROOT" \
     tools.rpent.python="$RPENT_PYTHON" \
     tools.rpent.output_dir="$EPISODE_DIR" \
     tools.rpent.remote_toolkit_endpoint=http://127.0.0.1:28490

This preset runs 150 optimizer updates, limits each episode to 50 model requests and each request to 16,384 output tokens, and saves a checkpoint every 10 updates. It keeps the base recipe's context, environment-step and timeout budgets, sampling, workflow and PPO settings. Start from the full SFT checkpoint with fresh output directories; saved-batch validation weights belong to their separate run.

``data.rollout_batch_size=256`` is the effective learner batch. ``data.sampling_batch_size=320`` consumes 320 dataset entries and starts all candidates. The coordinator selects the first 256 complete episodes it receives, including normal failures with reward zero. It cancels the remaining candidates, aborts their model requests and drains their replies and generation queues. Training starts after process, session and remote-owner cleanup passes. A real infrastructure failure follows the existing retry policy; exhausted retries stop the cohort. Dropped candidates do not become zero-reward training examples.

The dataset cursor advances by all 320 candidate entries; the learner uses exactly 256 episodes. Cohort audits retain candidate task IDs, completion order, selected IDs, policy versions, cancellation and admission receipts. Raw complete and partial traces remain in the episode directories. Report the candidate and retained sets separately because selection depends on completion time.

Oversampling applies only to synchronous agent PPO with group size one. Omitting ``data.sampling_batch_size`` uses the effective batch size and preserves the base behavior. Asynchronous and evaluation runs require equal sampling and effective batch sizes; use the separate evaluation config for the fixed 60-episode panel.

Resume from a complete checkpoint at a saved update boundary. Updates after that checkpoint need fresh collection on recovery; keep their original data for audit. Check actual optimizer progress and installed policy versions between saves, since updates 1–9 do not produce a checkpoint in this preset.

Reuse a Saved Batch
-------------------

To validate a memory fix, replay a complete, audited batch with its original policy,
tokenizer and processor. Keep its episode directories, immutable image assets and
manifest together. The manifest must contain the original policy-file hashes,
per-turn hashes and one terminal outcome per episode.

Run one update in a separate output directory:

.. code-block:: bash

   python examples/agent/rpent/replay_train.py \
     --config-path "$RLINF_ROOT/examples/agent/rpent/config" \
     --config-name replay_qwen3.5_4b_episode_sync_ppo \
     runner.output_dir="$REPLAY_RUN_DIR" \
     data.train_data_paths="[$SAVED_TASKS]" \
     rollout.model.model_path="$MODEL_PATH" \
     tools.rpent.repo_path="$RPENT_ROOT" \
     tools.rpent.python="$RPENT_PYTHON" \
     tools.rpent.output_dir="$REPLAY_EPISODE_DIR" \
     tools.rpent.remote_toolkit_endpoint=http://127.0.0.1:28490 \
     replay.source_dir="$SAVED_EPISODE_DIR" \
     replay.manifest="$SAVED_MANIFEST"

``SAVED_TASKS`` lists exactly the tasks in the audited batch; ``MODEL_PATH`` points
to that batch's original full policy. Match the batch size, deployment placement,
turn-merging limit and token budgets to the archived run. The replay uses the
production actor/critic update and initializes the rollout servers to reproduce
co-resident memory use. It reads sampled IDs, masks, logprobs and image references
from disk and issues zero model requests or environment actions. A successful run
writes ``replay-training-memory.json`` and a complete step-1 checkpoint under the
replay experiment directory. Keep these validation weights separate from a fresh
formal run.

.. warning::

   This NVIDIA recipe sets ``actor.env_vars.PYTORCH_CUDA_ALLOC_CONF`` and
   ``critic.env_vars.PYTORCH_CUDA_ALLOC_CONF`` to ``expandable_segments:True`` for
   varying sequence lengths. Scope this option to the learners. Setting it in
   the parent shell also affects SGLang and disables its TorchMemorySaver
   offload mechanism in the validated stack. Allocator settings cannot establish
   that a workload fits: verify that both the actor and critic complete an update with the original data.

Shared GPU Communication Memory
-------------------------------

The NVIDIA preset sets ``NCCL_MAX_CTAS=8`` in the actor, critic and rollout process environments to bound NCCL communication resources on shared GPUs. NCCL allocates device buffers outside PyTorch's allocator, and some connections allocate their buffers on the first collective. Offloading the model or clearing PyTorch's cache does not release these buffers.

Check GPU memory during the second rollout, after both the actor and critic have completed an update, weights have synchronized and the rollout KV cache has been restored. Compare total GPU memory use with the memory PyTorch reports as allocated and reserved. Leave enough GPU memory free for prefill and image processing. A training replay that stops before the rollout KV cache is restored cannot confirm that enough memory remains for the next rollout.

Changing the communication limit can affect throughput. Recheck update and synchronization times when changing placement or this environment variable.

Recover an Interrupted First Cohort
-----------------------------------

A fresh run can reuse its archived first cohort before collecting subsequent cohorts online. Use the original full SFT policy and the same dataset order, placement and budgets. The manifest must contain hashes for the admitted episodes and the policy files, plus every original candidate task. The selected set stays fixed; discarded candidates contribute no training data.

Append these arguments to the synchronous training command above:

.. code-block:: bash

   +replay.source_dir="$SAVED_EPISODE_DIR" \
   +replay.manifest="$SAVED_MANIFEST" \
   +replay.expected_version=0

The first update reads the archived sampled tokens and images, consumes all candidate dataset entries and uses exactly the admitted batch. Subsequent updates collect online episodes. Recovery from an update-boundary checkpoint skips this first-cohort replay. Keep the source manifest and its original sampling audit with the run, and account for the original rollout time separately from reconstruction time.

On shared GPUs, rollout KV restoration waits for weight installation, sender offload and allocator cleanup. Check that all rollout replicas publish the updated policy version before treating the next cohort as active.
