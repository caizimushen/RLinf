跨集群 RPent Episode PPO
========================

这个示例用同步 PPO 训练 Qwen3.5-4B planner，将 LIBERO-PRO 环境和 SAM3 放到独立 GPU 机器。训练使用 Goal、Spatial、Object 三个 suite 的完整 episode。\ ``episode`` workflow 显式调用感知工具，每批按顺序执行最多三个动作。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型

      Qwen3.5-4B 完整多模态 checkpoint

   .. grid-item-card:: 算法

      同步 PPO，独立 critic

   .. grid-item-card:: 任务

      LIBERO-PRO Goal、Spatial、Object

   .. grid-item-card:: 硬件

      两台各 8 卡的训练机器，一台 8 卡环境机器

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 项目
     - 设置
   * - 训练池
     - 30 个任务的全部原生初始状态；每个任务按状态值去重，保留原始 ID
   * - PPO batch
     - 256 条完整 episode，包含正常任务失败；每批更新一遍
   * - 回归面板
     - 30 个任务各测 seed 0、1，共 60 条；与训练池重叠
   * - 奖励
     - 环境原生成功为 1，正常任务失败为 0；每个 episode 只结算一次终局奖励

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 项目
     - 约定
   * - 工具
     - ``execute_actions``\ 、\ ``view_env``\ 、\ ``segment``\ 、\ ``back_project``\ 、\ ``finish``
   * - 图片
     - 显式返回 1024 像素双相机观测，图片与深度保留记录帧标识
   * - 历史
     - 保留 reasoning 和图片；达到 context 的 70% 或 80 条消息后压缩，保留 10 条
   * - 预算
     - 80 次模型请求、10000 环境步、262144 context tokens、每次最多输出 32768 tokens

planner 和图片预处理在训练机器运行，环境和 SAM3 调用走私有 RPC。训练图片经过传输和哈希校验，保存在两台训练机器共享的 episode 目录。工具反馈、图片和 prompt 的 loss mask 为 0；模型生成的 reasoning、tool-call 保留原始 token ID 和行为 logprob。

安装
----

两侧使用相同的已验证 RPent revision，RLinf checkout 须包含本示例。将 ``RLINF_ROOT`` 和 ``RPENT_ROOT`` 指向这两个源码目录。在两台训练机器安装 agentic Megatron/SGLang 环境：

.. code-block:: bash

   cd "$RLINF_ROOT"
   UV_TORCH_BACKEND=cu128 UNINSTALL_FA4=1 bash requirements/install.sh agentic \
     --engine sglang --python 3.12.12 --torch 2.11.0 \
     --sglang 0.5.12.post1 --transformers 5.6.1 --install-rlinf
   source .venv/bin/activate
   uv pip install -e "$RPENT_ROOT"

两台训练机器的 Python、Torch、SGLang、Transformers、Megatron-Core、Bridge 版本必须一致。加载权重前验证 CUDA、Transformer Engine、FlashAttention 能正确导入。模型和环境安装的前置条件见 :doc:`rpent_libero_rl_setup`。

环境机器使用独立的仿真／SAM3 依赖文件。先导出 ``RLINF_ROOT``\ ，让安装使用配对 checkout；这个依赖文件不安装 OpenPi：

.. code-block:: bash

   cd "$RPENT_ROOT"
   export RLINF_ROOT
   uv venv --python 3.12 .env-ppo
   source .env-ppo/bin/activate
   uv pip install --prerelease=allow --torch-backend cu128 \
     -r requirements/episode-ppo-env.txt
   export PYTHONPATH="$RPENT_ROOT:$RLINF_ROOT${PYTHONPATH:+:$PYTHONPATH}"

按 RPent 的 ``tools/rl/INSTALL.md`` 安装 LIBERO-PRO 资产并配置路径，导出 ``LIBERO_TYPE=pro``\ 、\ ``LIBERO_CONFIG_PATH``\ 、\ ``MUJOCO_GL=egl``\ 、\ ``PYOPENGL_PLATFORM=egl``\ 。每张 GPU 启动两个 SAM3，再启动统一 pool，使用 ``rpent.robots.components.sam3_server`` 和 ``sam3_pool``\ ；完整命令见 RPent 的 ``tools/rl/WORKFLOWS.md``\ 。之后启动远程后端：

.. code-block:: bash

   cd "$RPENT_ROOT"
   python -m robots.rl.remote_manager \
     --output-dir "$ENV_OUTPUT" \
     --gpu-slots 0:32,1:32,2:32,3:32,4:32,5:32,6:32,7:32 \
     --max-initializing-per-gpu 2 --sam3-endpoint "$SAM_POOL" \
     --host 127.0.0.1 --port 28490

``ENV_OUTPUT`` 保留所有后端尝试，\ ``SAM_POOL`` 是环境机器本地的 pool URL。训练前，用真实动作、感知突发请求和至少五分钟驻留验证 256 环境、16 SAM3 的布局。slot 数只是上限，不能证明容量足够。

在环境服务器的两个终端中分别运行反向转发，将监听端口映射到两台训练机器。提前配置示例 SSH 别名，并完成认证：

.. code-block:: bash

   ssh -NT -o ExitOnForwardFailure=yes \
     -R 127.0.0.1:28490:127.0.0.1:28490 <training-host-0>
   ssh -NT -o ExitOnForwardFailure=yes \
     -R 127.0.0.1:28490:127.0.0.1:28490 <training-host-1>

在环境服务器上持续运行并监控两条转发，私钥留在服务器，并固定 host key。两台训练机器通过私有 Ray/NCCL 网络通信，使用共享文件系统。按 :doc:`/rst_source/guides/multi_node` 启动 Ray，每台机器先设置独立的 ``RLINF_NODE_RANK``\ 。环境服务器不加入这个 Ray 集群。

下载模型
--------

将原始完整多模态 checkpoint 下载到两台训练机器共享的模型目录：

.. code-block:: bash

   hf download Qwen/Qwen3.5-4B --local-dir "$MODEL_PATH"

两台训练机器的 ``MODEL_PATH`` 必须指向相同权重、tokenizer 和 processor。本示例开启 thinking，4B 模板不支持 reasoning-effort 档位。

运行
----

在环境机器使用本机的 LIBERO 配置，生成原生训练状态池和固定回归面板：

.. code-block:: bash

   python "$RLINF_ROOT/examples/agent/rpent/prepare_tasks.py" \
     --workflow episode --libero-type pro --seeds all --split train \
     --libero-config-path "$LIBERO_CONFIG_PATH" --output-dir "$TASK_DIR"
   python "$RLINF_ROOT/examples/agent/rpent/prepare_tasks.py" \
     --workflow episode --libero-type pro --seeds 0,1 --split eval \
     --libero-config-path "$LIBERO_CONFIG_PATH" --output-dir "$TASK_DIR"

将 ``TASK_DIR`` 复制到训练机器共享的文件系统。两个 manifest 记录原始状态 ID、状态哈希和数据集哈希，须与数据一起保留。这 60 条只检查训练池状态上的回归，不能当作留出集泛化评估。

在 Ray head 上运行以下命令，传入各目录的实际路径：

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

``RPENT_PYTHON`` 在每台训练机器运行 planner。两侧 learner 必须共享 ``EPISODE_DIR``\ ，并持续保留该目录，以便从 checkpoint 恢复。actor/critic 使用 TP2、CP1、DP8，rollout 使用 16 个 TP1 server；16 个 agent worker 分别固定对应一个 server，每个 worker 有 16 个 episode slot。

先用独立目录设置 ``data.rollout_batch_size=8``\ 、\ ``runner.max_steps=2``\ ，在完整更新边界保存并用 ``runner.resume_dir`` 恢复一次。之后从原始 checkpoint 开始验收，每次更新采集 256 条 episode，共更新三次，使用全新的 ``RUN_DIR`` 和 ``EPISODE_DIR``\ 。小批也保持 80 次请求和 32768 输出预算。

固定 60 条评估沿用相同的 workflow 和采样设置：

.. code-block:: bash

   bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b_episode_sync \
     runner.output_dir="$RUN_DIR" \
     data.val_data_paths="[$TASK_DIR/eval.jsonl]" \
     rollout.model.model_path="$MODEL_PATH" \
     tools.rpent.repo_path="$RPENT_ROOT" \
     tools.rpent.python="$RPENT_PYTHON" \
     tools.rpent.output_dir="$EPISODE_DIR" \
     tools.rpent.remote_toolkit_endpoint=http://127.0.0.1:28490

初始评估把 ``MODEL_PATH`` 指向原始 checkpoint，最终评估指向 ``checkpoint.py export`` 导出的 policy。两次评估分别使用独立的结果和 episode 目录。评估会把 60 条完整分配给 16 个 worker，不增补或丢弃 episode。

可视化与结果
------------

本示例设置 temperature 1、top-p 1、关闭 top-k、presence penalty 0、repetition penalty 1。actor/critic 学习率为 1e-6/5e-6，gamma/lambda 为 1，clip 为 0.2，每批更新一遍。这是验收的起步配置，不是已证实的最优配置。

检查原生成功率、完整 episode 数、固定 policy version、actor/critic loss 和梯度、TIS 偏差，以及模型、感知、通信耗时。每条 episode 保留 HTTP 记录、采样 token 审计和有序训练图片。正常任务失败，以及未取得环境原生成功的 context 耗尽、输出截断等预算终止，均按零奖励终局处理。基础设施失败保留尝试，最多额外重试一次；重试耗尽停止该批。

报告训练结果前，用相同 IDs、图片网格和 mask 对比 HF、Megatron、SGLang 的选中 token logprob。确认语言参数和 critic 确实更新，冻结视觉参数没有改变，并核对全部 rollout server 的权重版本。初始化和第三次更新后各运行同一组 60 条 episode，按相同 task/seed 对比耗时。

打过补丁的 SGLang 后端用 FP32 保存 Qwen3.5 GDN 的输出归一化权重，导出权重时也用 FP32 计算 ``1 + actor_gamma``\ ，保留 actor 的有效缩放值。检查数值一致性时，HF 参考模型也要保留这些 checkpoint 参数的精度。

三次更新只提供接入与稳定性证据，不能证明 PPO 配置最优或泛化提升。


在 32 张 GPU 上运行同步超发
---------------------------

320 候选配置使用四台各 8 卡的模型机器，以及一台 8 卡环境机器。actor 和 critic 使用 TP2、CP1、DP16；32 个 TP1 rollout server 各对应一个 agent worker，每个 worker 有 10 个 episode slot。每张环境 GPU 放 40 个环境和两个 SAM3。先用真实动作、双相机感知、五分钟驻留和自有进程清理验证容量。

环境后端空闲后，使用独立输出目录，将槽位上限设为每卡 40：

.. code-block:: bash

   python -m robots.rl.remote_manager \
     --output-dir "$ENV_OUTPUT" \
     --gpu-slots 0:40,1:40,2:40,3:40,4:40,5:40,6:40,7:40 \
     --max-initializing-per-gpu 2 --sam3-endpoint "$SAM_POOL" \
     --host 127.0.0.1 --port 28490

四台模型机器都要有经过认证的反向转发，并验证私有 Ray/NCCL 网络上的四节点、32 GPU。配对的 RPent revision 必须支持在环境启动期间按 attempt 归属取消。被取消的 attempt 仍保留后端数据和清理回执。

沿用基础示例的路径变量，启动以下配置：

.. code-block:: bash

   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_sync_oversample320 \
     runner.output_dir="$RUN_DIR" \
     data.train_data_paths="[$TASK_DIR/train.jsonl]" \
     rollout.model.model_path="$MODEL_PATH" \
     tools.rpent.repo_path="$RPENT_ROOT" \
     tools.rpent.python="$RPENT_PYTHON" \
     tools.rpent.output_dir="$EPISODE_DIR" \
     tools.rpent.remote_toolkit_endpoint=http://127.0.0.1:28490

这个配置执行 150 次 optimizer 更新，每条 episode 最多请求模型 50 次，每次输出最多 16384 tokens，每 10 次更新保存 checkpoint。context、环境步数、超时预算、采样、workflow 和 PPO 设置沿用基础配置。从完整 SFT checkpoint 初始化，并使用全新输出目录；保存批次回放产生的验证权重留在独立运行中。

\ ``data.rollout_batch_size=256`` 指有效 learner batch。\ ``data.sampling_batch_size=320`` 会读取 320 条数据，并启动全部候选。协调器选取最先收到的 256 条完整 episode，包含奖励为零的正常失败。随后取消其余候选、中止模型请求，并收回响应、清理生成队列。进程、session 和远端 owner 清理通过后才开始训练。真实基础设施故障沿用原重试规则，重试耗尽就停止该 cohort。丢弃候选不会变成零奖励训练样本。

数据游标按全部 320 条候选推进，learner 恰好使用 256 条 episode。cohort 审计记录候选任务 ID、完成顺序、选中 ID、policy version、取消和投递回执。完整与部分原始轨迹均保留在 episode 目录。选择依赖完成时间，报告时须分别统计候选集合与保留集合。

超发只用于 group size 为 1 的同步 agent PPO。省略 \ ``data.sampling_batch_size`` 会使用有效 batch size，沿用基础行为。异步和评估要求采样条数等于有效 batch size；固定 60 条面板使用独立评估配置。

恢复时使用已完整保存的更新边界 checkpoint。之后尚未保存的更新需要重新采集，原数据保留用于审计。两次保存之间应检查实际 optimizer 进度和已安装的 policy version；这个配置的第 1～9 次更新不会生成 checkpoint。

复用保存的批次
--------------

验证显存修复时，使用完整、已审计的批次及其原始 policy、tokenizer 和 processor。保留 episode 目录、不可变图片和 manifest。manifest 必须记录原始模型文件哈希、逐 turn 哈希，以及每条 episode 的终局结果。

在独立输出目录中执行一次更新：

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

``SAVED_TASKS`` 须恰好包含已审计批次的任务，\ ``MODEL_PATH`` 指向该批次的原始完整 policy。batch size、GPU 布局、turn 合并上限和 token 预算须与原始运行一致。回放沿用正式 actor/critic 更新，并初始化 rollout server，复现 learner 与 rollout server 同时驻留时的显存占用。采样 ID、mask、logprob 和图片引用均从磁盘读取；模型请求和环境动作数均为零。成功后，回放实验目录中会生成 ``replay-training-memory.json`` 和完整的 step-1 checkpoint。回放产生的验证权重须与新正式实验的初始化权重分开保留。

.. warning::

   这个 NVIDIA 配置将 ``actor.env_vars.PYTORCH_CUDA_ALLOC_CONF`` 和 ``critic.env_vars.PYTORCH_CUDA_ALLOC_CONF`` 设为 ``expandable_segments:True``\ ，适应变化的序列长度。只给 learner 设置此选项。在父 shell 中设置会影响 SGLang，并在已验证的依赖栈中禁用 TorchMemorySaver 卸载。分配器配置不能证明容量足够，须用原始完整数据验证 actor 和 critic 都能完成更新。

共享 GPU 的通信显存
-------------------

这个 NVIDIA 配置在 actor、critic 和 rollout 的进程环境中设置 ``NCCL_MAX_CTAS=8``\ ，限制共卡运行的 NCCL 通信资源。NCCL 在 PyTorch 分配器之外分配 GPU 缓冲区。有些连接直到第一次集合通信时才分配缓冲区。卸载模型或清理 PyTorch 缓存都不会释放这些缓冲区。

actor 和 critic 完成一次更新、权重同步完毕、rollout 的 KV 缓存恢复后，在第二轮 rollout 运行时检查显存。将整卡占用与 PyTorch 的 allocated、reserved 指标对照，给预填充和图片处理留出空间。训练回放如果停在 rollout 的 KV 缓存恢复前，就无法确认下一轮 rollout 的显存是否够用。

通信上限可能影响吞吐。调整 GPU 布局或这个环境变量后，重新检查更新与权重同步的耗时。

恢复中断的首批数据
------------------

从原始完整 SFT policy 重新启动后，可以先用保存的首批数据完成更新，再继续在线采集。数据顺序、GPU 布局和预算须与原运行一致。manifest 需要记录已选 episode 和模型文件的哈希，以及全部原始候选任务。恢复后沿用原选中集合；被丢弃的候选不进入训练。

在上面的同步训练命令后追加以下参数：

.. code-block:: bash

   +replay.source_dir="$SAVED_EPISODE_DIR" \
   +replay.manifest="$SAVED_MANIFEST" \
   +replay.expected_version=0

第一次更新读取保存的采样 token 和图片，数据游标按全部候选推进，learner 只使用原选中的批次。此后恢复在线采集。从更新边界 checkpoint 恢复会跳过首批回放。保留 manifest 及其原始采样审计，分别记录原 rollout 耗时和数据重建耗时。

共享 GPU 上的 rollout KV 恢复要等待权重安装、发送端卸载和显存缓存清理。确认全部 rollout 副本都发布更新后的 policy version，再认定下一批已经开始。
