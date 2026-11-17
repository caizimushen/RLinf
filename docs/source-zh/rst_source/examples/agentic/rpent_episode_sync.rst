RPent Episode PPO 与 GRPO
=========================

训练 Qwen3.5-4B 工具调用规划器，由远端 LIBERO-PRO 环境执行工具调用。rollout、训练和评测使用同一份完整多模态 SFT checkpoint，并保持 SFT 时的 RPent 工作流。本例采用 ``episode``：显式感知，每次最多顺序执行三个动作。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型

      Qwen3.5-4B；冻结视觉编码器和 projector

   .. grid-item-card:: 算法

      同步 GRPO，或带独立 critic 的 PPO

   .. grid-item-card:: 任务

      LIBERO-PRO Goal、Spatial、Object

   .. grid-item-card:: 硬件

      四台各 8 张 H100 的模型主机，一台 8 张 A100 的环境主机

任务
~~~~

GRPO 配置执行 150 次更新，每次按完整组选择训练数据。

.. list-table::
   :header-rows: 1

   * - 项目
     - 配置
   * - 训练池
     - 30 个任务的原生状态；同一任务内去重，保留原始状态编号
   * - GRPO batch
     - 32 组 × 8 条，共 256 条 episode；同组使用同一任务和原生初始状态
   * - 超发
     - 同时启动 40 组 × 8 条，选最先完整结束的 32 组；取消剩余 64 条，确认生成请求已停止，并释放全部 320 个环境后开始训练
   * - 奖励
     - 原生成功为 1，正常失败为 0；全 0、全 1 组均保留
   * - 训练池回归评测集
     - 30 个任务的 seed 0、1，共 60 条训练池 episode

观测与动作
~~~~~~~~~~

配置保持 SFT 的 prompt、工具 schema 和历史保留规则。

.. list-table::
   :header-rows: 1

   * - 项目
     - 约定
   * - 观测
     - 显式获取 1024 像素相机图像；首次观测前的 turn 可以没有图像
   * - 动作
     - ``execute_actions`` 最多顺序执行三个动作；感知工具为 ``view_env``、``segment``、``back_project``，输出工具为 ``finish``
   * - 奖励
     - 以环境成功信号为准；规划器调用 ``finish`` 不代表任务成功
   * - Prompt
     - thinking 开启，不发送 effort 档位；保留推理和图像历史；达到上下文的 70% 或 80 条消息时压缩，保留 10 条
   * - 预算
     - 上下文上限为 262144 个 token，每次最多输出 16384 个 token；最多 50 次请求、10000 环境步、规划器 7200 秒、collector 14400 秒

GRPO 如何训练
-------------

每组按奖励均值和标准差计算 advantage，保留奖励相同的组。GRPO 不使用 critic，也不再做第二次全 batch advantage 归一化。配置使用 group、agent、turn 三级 loss scaling，按有效 token 求均值，clip 下界 0.2、上界 0.28；学习率 1e-6、warmup 为 0，KL 和 entropy 系数为 0，梯度裁剪为 1，每个 batch 学习一遍。实际采样的推理和 tool-call token 参与学习；prompt、工具反馈、图像和 padding 的 mask 为 0。

按完成速度选择完整组可能造成选择偏差。用 sampling manifest 比较选入和取消的组；不同训练 batch 的成功率不能替代固定条件下的 policy 对比。

安装
----

模型主机需要 Linux x86-64、Git、``uv``、C++ 编译器、Rust、``patch`` 和 CUDA 13 开发工具链。FA4 训练路径需要 H100 或更新的 GPU，以及兼容 CUDA 13 wheel 的驱动。环境主机另外准备 EGL/OpenGL 和下文的 CUDA 12.8 仿真环境。

已有 RLinf ``reason`` 镜像可以提供系统工具，但其中默认的 Python 环境不包含这套 TokenSpace 依赖。可以在 CUDA 13 开发镜像内，或已准备好工具链的主机上，安装独立环境。镜像构建选项见 ``docker/README.md``；下文用安装脚本明确依赖，不依赖未记录版本的现有镜像。

选择共享目录，克隆两个仓库的 ``rpent-ppo`` 分支。部署实验时记录具体 commit：

.. code-block:: bash

   export STACK_ROOT="$HOME/rpent-stack"
   mkdir -p "$STACK_ROOT/repos"
   git clone --branch rpent-ppo https://github.com/zhexuanxu/RLinf.git "$STACK_ROOT/repos/RLinf"
   git clone --branch rpent-ppo https://github.com/zhexuanxu/RPent.git "$STACK_ROOT/repos/RPent"
   export RLINF_ROOT="$STACK_ROOT/repos/RLinf"
   export RPENT_ROOT="$STACK_ROOT/repos/RPent"
   git -C "$RLINF_ROOT" rev-parse HEAD
   git -C "$RPENT_ROOT" rev-parse HEAD

每台模型主机都安装独立的 TokenSpace 环境：

.. code-block:: bash

   cd "$RLINF_ROOT"
   UV_TORCH_BACKEND=cu130 UNINSTALL_FA4=0 bash requirements/install.sh agentic \
     --engine sglang --sglang token-space --torch 2.14.1 \
     --transformers 5.19.0 --python 3.12.12 \
     --venv "$STACK_ROOT/envs/model" --install-rlinf
   source "$STACK_ROOT/envs/model/bin/activate"
   uv pip freeze > "$STACK_ROOT/model-packages.txt"

该配置固定 SGLang commit ``18a44326f6de5134fe921927d760d81f8ee48fc9``、Torch 2.14.1、Transformers 5.19.0，并安装 Transformer Engine 2.17、Megatron-Bridge 0.5、Megatron 0.18 源码及 CPU offload checkpoint 修补。自行设置 ``MEGATRON_PATH`` 时，使用 commit ``dc0ee41ada177d4c74ba46bb501bba5967264815``，并手动应用以下补丁：

.. code-block:: text

   requirements/agentic/megatron-core-r0.18.0-cpu-offload.patch

安装器还会应用 `Transformer Engine PR 3540 <https://github.com/NVIDIA/TransformerEngine/pull/3540>`__ 中的 FA4 窗口参数适配。该修补只改 FA4 的 Python 调用边界，保留原生 attention 内核。FA2 和 FA4 应放在不同环境中。各节点必须保留完整 Python 环境、Python 源码、依赖源码树和 JIT 头文件；只复制部分 site-packages 无法满足需要读取源码的编译器。不要替换或清理运行中作业使用的环境。

每台模型主机另装 CPU planner 环境。它运行 RPent 模型客户端，通过 RPC 调用远端工具，并用本地 checkpoint processor 统计 prompt token；不加载规划器权重或模拟器：

.. code-block:: bash

   cd "$RPENT_ROOT"
   uv venv --python 3.12 "$STACK_ROOT/envs/planner"
   uv pip install --python "$STACK_ROOT/envs/planner/bin/python" \
     --prerelease=allow --torch-backend cpu -r requirements/episode-planner.txt

将 ``tools.rpent.python`` 指向 ``$STACK_ROOT/envs/planner/bin/python``。启动 Ray 前重新激活 model 环境。

在环境主机安装仿真和 SAM3：

.. code-block:: bash

   cd "$RPENT_ROOT"
   export RLINF_ROOT
   uv venv --python 3.12 "$STACK_ROOT/envs/environment"
   source "$STACK_ROOT/envs/environment/bin/activate"
   uv pip install --prerelease=allow --torch-backend cu128 \
     -r requirements/episode-ppo-env.txt
   export PYTHONPATH="$RPENT_ROOT:$RLINF_ROOT${PYTHONPATH:+:$PYTHONPATH}"
   export LIBERO_TYPE=pro MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

``episode-ppo-env.txt`` 同时用于 PPO、GRPO 的 episode 环境，包含 LIBERO/LIBERO-PRO、SAM3、MuJoCo、robosuite 和 RPC 依赖。按 RPent 的 ``tools/rl/INSTALL.md`` 下载 LIBERO-PRO 资源，生成 ``LIBERO_CONFIG_PATH/config.yaml``。环境主机的 Transformers 4.57 与模型主机的 Transformers 5.19 分开安装。

下载模型
--------

启动服务前准备以下资源。安装 Python 包不会自动下载模型权重或场景资源。

.. list-table::
   :header-rows: 1

   * - 资源
     - 使用方
     - 准备方式
   * - 完整 Qwen3.5-4B SFT 目录
     - RLinf rollout 和 learner
     - 权重、config、tokenizer、chat template、processor；各模型主机使用相同绝对路径
   * - SAM3 checkpoint
     - 环境主机的 SAM3 服务
     - 申请 ``facebook/sam3`` 访问权限，下载 ``sam3.pt``
   * - LIBERO-PRO 资源
     - 环境 worker
     - 场景资源，以及包内 BDDL 文件和原生初始状态

对比 SFT 时，将 ``MODEL_PATH`` 指向原始完整多模态 SFT 目录，目录中需包含 processor 和 tokenizer 文件：

.. code-block:: bash

   export MODEL_PATH=/absolute/path/to/full-sft-checkpoint

基础模型实验则单独下载 Qwen 完整模型：

.. code-block:: bash

   export MODEL_PATH="$STACK_ROOT/models/Qwen3.5-4B"
   hf download Qwen/Qwen3.5-4B --local-dir "$MODEL_PATH"

在环境主机下载 SAM3 checkpoint：

.. code-block:: bash

   export SAM3_CHECKPOINT_PATH="$STACK_ROOT/models/sam3/sam3.pt"
   hf download facebook/sam3 sam3.pt --local-dir "$(dirname "$SAM3_CHECKPOINT_PATH")"

直接控制的 ``episode`` 工作流不使用 Pi0.5 权重。Pi0.5 属于 :doc:`原始规划器示例 <rpent>`；切换到该示例会改变动作工具。使用 episode 工作流做过 SFT 的模型应保持 ``workflow=episode``。

运行
----

启动 SAM3 和环境服务
~~~~~~~~~~~~~~~~~~~~

每张环境 GPU 启动两个 SAM3 实例，分别使用独立端口。例如第一张卡的两个实例：

.. code-block:: bash

   python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8114
   python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8115

在独立终端或服务监督器下运行，GPU 1–7 也各启动两个实例并分配不同端口。将全部 16 个地址传给 ``rpent.robots.components.sam3_pool --backends``；完整服务示例见 RPent 的 ``tools/rl/WORKFLOWS.md``。把 pool URL 写入 ``SAM_POOL``，再启动环境 backend：

.. code-block:: bash

   python -m robots.rl.remote_manager \
     --output-dir "$STACK_ROOT/environment-runs" \
     --gpu-slots 0:40,1:40,2:40,3:40,4:40,5:40,6:40,7:40 \
     --max-initializing-per-gpu 2 --sam3-endpoint "$SAM_POOL" \
     --host 127.0.0.1 --port 28490

backend 提供 320 个槽位，采样中已结束的 episode 会逐步释放槽位，因此 ready 的瞬时数量可以低于 320。学习阶段应已释放全部环境，SAM3 仍驻留显存。实际容量还取决于 GPU 和主存余量。

用已认证的私有路由将该 loopback 服务提供给每台模型主机。若使用 SSH 反向转发，在环境服务器上为每台模型主机分别运行，并固定主机密钥：

.. code-block:: bash

   ssh -NT -o ExitOnForwardFailure=yes \
     -R 127.0.0.1:28490:127.0.0.1:28490 model-host-0

在环境服务器上监督这些转发进程。四台模型主机加入同一私有 Ray 集群，并分别设置 ``RLINF_NODE_RANK``，参见 :doc:`/rst_source/guides/multi_node`。环境主机不加入 Ray。启动 Ray 前激活一致的 model 环境并导出路径；训练入口只在 Ray head 执行一次。

准备任务和路径
~~~~~~~~~~~~~~

在已安装 LIBERO-PRO 的环境主机上生成任务文件：

.. code-block:: bash

   export TASK_DIR="$STACK_ROOT/tasks"
   python "$RLINF_ROOT/examples/agent/rpent/prepare_tasks.py" \
     --workflow episode --libero-type pro --seeds all --split train \
     --libero-config-path "$LIBERO_CONFIG_PATH" --output-dir "$TASK_DIR"
   python "$RLINF_ROOT/examples/agent/rpent/prepare_tasks.py" \
     --workflow episode --libero-type pro --seeds 0,1 --split eval \
     --libero-config-path "$LIBERO_CONFIG_PATH" --output-dir "$TASK_DIR"

将任务目录和 manifest 复制到模型主机的共享文件系统。包含 60 条 episode 的评测集是训练池回归，不是泛化评测。

复制路径模板，在仓库外保存部署配置：

.. code-block:: bash

   export CONFIG_DIR="$STACK_ROOT/config"
   mkdir -p "$CONFIG_DIR/deployment"
   cp "$RLINF_ROOT/examples/agent/rpent/config/deployment/episode.yaml" \
     "$CONFIG_DIR/deployment/local.yaml"

填完 ``local.yaml`` 中全部 ``???``。在 ``data`` 下按列表填写刚生成的训练、评测 JSONL 文件：

.. code-block:: yaml

   data:
     train_data_paths: ["/data/tasks/train.jsonl"]
     val_data_paths: ["/data/tasks/eval.jsonl"]

将以下模型路径都指向同一份完整 SFT：

.. code-block:: text

   rollout.model.model_path
   actor.tokenizer.tokenizer_model
   critic.tokenizer.tokenizer_model
   actor.megatron.ckpt_convertor.hf_model_path
   critic.megatron.ckpt_convertor.hf_model_path

``tools.rpent.python`` 指向 planner 解释器，``repo_path`` 指向匹配的 RPent 源码。episode 输出和预处理 cache 在各模型主机上必须有相同绝对路径；只要轨迹或 checkpoint 仍引用图片，就保留这些图片文件。

启动训练
~~~~~~~~

在 Ray head 启动 GRPO：

.. code-block:: bash

   cd "$RLINF_ROOT"
   source "$STACK_ROOT/envs/model/bin/activate"
   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_sync_grpo \
     --config-dir "$CONFIG_DIR" +deployment@_global_=local

actor 使用 TP2/CP1/DP16，rollout 为 32 个 TP1 副本。actor、critic、rollout 保持 ``NCCL_MAX_CTAS=8``，SGLang 显存比例为 0.55。该比例不是共用 GPU 的总显存上限；对比实验时保持相同的 placement 和显存设置。

PPO 可以共用相同的预处理环境，启动时选择 PPO 配置和 TokenSpace 配置：

.. code-block:: bash

   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_sync_oversample320 \
     +preprocessing@_global_=token_space \
     --config-dir "$CONFIG_DIR" +deployment@_global_=local

该 PPO 配置同时启动 320 个独立候选，选最先完整结束的 256 条，使用独立 critic 和按完整 episode 时间顺序计算的 GAE，不使用 GRPO 组内归一化。

新环境首次运行时，先核实完整 optimizer 更新、权重同步和下一批真实在线 episode。cohort 完成或单个 TensorBoard tag 不代表一次更新已完成。新环境的 GPU 数值一致性及其他驱动、内核版本的兼容性仍需验证。

恢复、导出和评测
~~~~~~~~~~~~~~~~

通过 ``runner.resume_dir`` 从相同配置的完整 checkpoint 恢复。按角色检查模型和 optimizer 分片、scheduler/RNG，以及原子发布的数据状态。数据游标属于当前 epoch，不必等于全局更新数。保留中断的原始 episode；超过最近完整 checkpoint 的内存更新不能视为可恢复。参见 :doc:`/rst_source/guides/resume`。

完成第 150 次更新后，将训练得到的语言参数导出到原始完整多模态模型：

.. code-block:: bash

   python examples/agent/rpent/checkpoint.py export \
     --checkpoint "$CHECKPOINT_150" --vlm "$MODEL_PATH" --output "$EXPORT_PATH"

使用匹配的 model 环境，在准备好的训练池回归评测集上评测导出权重，并沿用训练的 TokenSpace rollout 设置：

.. code-block:: bash

   bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b_episode_token_space \
     --config-dir "$CONFIG_DIR" +deployment@_global_=local \
     rollout.model.model_path="$EXPORT_PATH"

评测使用独立输出目录；保留原始 SFT 和各次 checkpoint。正常失败以及输出、上下文、轮数、时间预算耗尽均为零奖励；只有分类明确的基础设施错误使用有界重试。第二次基础设施尝试仍失败时，应停止接收该批数据并检查原因。

可视化与结果
------------

TensorBoard 记录 optimizer loss、梯度范数、学习率、采样成功率、TIS 诊断和更新耗时。sampling manifest 记录选入组、取消成员和清理回执；成功率分母使用实际选入的 episode，不使用目录数量。episode 文件保留请求、结果、图片引用和训练 turn。

episode 训练默认关闭完整 HTTP 请求、响应和压缩前后输入的抓取。排查 planner 传输问题时设置 ``tools.rpent.record_api_trace=true``；直接运行 RPent 时使用 ``--record-api-trace``。开关不改变 prompt、压缩决策、token mask 或训练记录。

分别报告任务失败、基础设施错误、选入和取消组构成、token/turn 成本，以及准确的源码和依赖版本。60 条训练池回归不能当作独立测试集上的结果。

纯图像控制
----------

pear 配置在每次动作后提供双相机 RGB 和实测机器人状态，沿用 GRPO 算法及每批 32 组、每组 8 条、超发 40 组的设置。仅提供 ``move_to`` 和 ``set_gripper`` 两个工具，``move_to`` 使用绝对位姿。环境判定任务成功或预算耗尽后，episode 自动结束。非法动作回复按正常失败处理，不再请求模型纠正。

沿用上文的环境、完整多模态 planner checkpoint、任务清单和部署配置。启动相同版本的 RPent backend；仅运行 pear 的 backend 不需要 SAM3 端点或权重。在 Ray head 启动：

.. code-block:: bash

   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_pear_sync_grpo \
     --config-dir "$CONFIG_DIR" +deployment@_global_=local

配置默认 ``tools.rpent.world_cord=general``：原点在安装基座中心，x 向前、y 向左、z 向上，抓取 TCP 是两指指垫中心的实测中点，初始观测给出桌面相对基座的高度。现有 LIBERO 感知工作流也会按此设置返回几何坐标。设为 ``tools.rpent.world_cord=base`` 可保留仿真世界坐标和原控制点。

thinking 默认开启。添加 ``agentloop.chat_template_kwargs.enable_thinking=false`` 后，要求模型直接调用工具，不输出推理文本。RPent CLI 对应 ``--reasoning`` 和 ``--no-reasoning``。切换工作流或坐标系时，使用独立运行目录，并固定源码版本；原有 SFT 工作流仍保留默认行为。
