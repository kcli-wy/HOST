# HumanGen 人类–机器人视频配对转换

本目录将 `/AOSS/HumanGen` 转为现有 `AlignmentDataset` 可读取的 episode。
HumanGen 的输入输出根目录、数据源和训练覆盖参数集中在 `config.json`；
相对路径以该配置文件所在目录为基准。
生成文件默认位于本目录的 `output/`，已加入忽略规则。
只读取已展开的机器人 episode、配对索引和人类 MP4；不需要 `human_latents/`
或重新解压 `part-*.tar.zst`。

数据规模、帧数与时长、来源复用、验证集划分风险和训练采样分析见
[`ANALYSIS_REPORT.md`](./ANALYSIS_REPORT.md)。`analyze.py` 可重算统计与图表。

## 配对依据

- 读取 `icl_configs/ICL_config_{dataset}.json` 的 `samples`，使用已有明确配对，不按任务名称重新组合。
- agibot、interna1、oxe、robocoin、robomind 的机器人视频已裁剪、重新编号。
  通过各任务 `meta/episodes.jsonl` 的 `icl.manifest_position` 定位新 episode，并核对
  `run`、`sample`、`human_video_path` 及 `source.video_rel_path`。
  同一编号有多个导出版本时，再按原始帧区间精确筛选；仍不唯一则跳过并报告。
  对缺少 `icl` 身份信息的记录，只接受唯一且精确的 `source.video_rel_path` 匹配。
- RoboTwin 保留原始编号，使用配对索引的机器人视频路径，并通过本地 episode 元数据核对帧数。
- 人类路径去掉旧的 `icl_raw_data/` 前缀，映射到当前 HumanGen 根目录。
- 已裁剪的视频直接使用，不重复套用原始帧区间。旧帧区间保留在 `pairs.jsonl` 供追溯。

默认使用配对索引指定的机器人视角（通常是 head/front/top），两侧统一命名为 `images.mp4`。
当前 alignment 加载器会把单视角循环补齐到三槽位；这不代表存在三路人类相机。

## 需要运行的文件

以下命令均从 HOST 仓库根目录执行，使用现有环境，无需重新安装依赖。

### 1. 转换全部数据：`prepare.py`

```bash
/data/250010175/miniforge3/envs/HOST_Alignment/bin/python \
  alignment/humangen_pairs/prepare.py
```

脚本检查视频存在、Decord 可读首帧、实际帧数与机器人元数据一致，以及两侧均不少于
`training.num_frames`（默认 24 帧）。不合格记录进入 `skipped.jsonl`，统计写入 `summary.json`。
Decord 可读的视频通过绝对路径软链接复用，不修改 `/AOSS/HumanGen`。
当前 RoboTwin 机器人视频为 AV1，而环境中的 Decord 不支持该解码器；脚本通过现有 PyAV
在 `output/compatible_videos/` 生成 H.264 兼容副本（libx264、CRF 18、veryfast），
保留帧数、分辨率及平均帧率。原视频路径记录在 `pairs.jsonl` 的 `robot_original_source`
或 `human_original_source`，训练链接指向兼容副本。其他解码错误不会被当成 AV1 静默修复。
重复运行可复用相同软链接；目标指向其他视频或存在同名普通视频时会报错。
以当次生成的索引为准；重新配置后未再入选的旧 episode 目录不会自动删除。

如需先小规模检查，输出到独立目录，避免覆盖完整索引：

```bash
/data/250010175/miniforge3/envs/HOST_Alignment/bin/python \
  alignment/humangen_pairs/prepare.py \
  --limit-per-dataset 3 --output-root alignment/humangen_pairs/smoke_output
```

`--limit-per-dataset` 限制检查原始记录数，过滤后可用配对数可能更少。

### 2. 验证训练读取：`validate.py`

```bash
/data/250010175/miniforge3/envs/HOST_Alignment/bin/python \
  alignment/humangen_pairs/validate.py --with-processor
```

验证全部索引结构，默认每个数据源抽取两对，直接调用真实加载器的 train/eval 取样逻辑，
核对配对身份和帧数，并实际解码所采样帧。`--with-processor` 额外用本地 Qwen processor
和训练 collator 构建一个真实输入 batch；不加载 8B 权重。结果写入 `validation_report.json`。
小规模输出可加 `--output-root alignment/humangen_pairs/smoke_output`。

### 3. 按主 README 启动训练：`train_scripts/run_ds.sh`

数据已转换时，直接使用原训练入口（以下命令从 HOST 仓库根目录执行）：

```bash
conda activate HOST_Alignment
cd alignment
VIDEO_PATHS=humangen_pairs/output/HumanGen_video_paths.json bash train_scripts/run_ds.sh
```

`alignment/config.py` 默认使用生成的相机映射，并关闭关节输入。训练仍由原有
`train.py`、`AlignmentDataset` 和 DeepSpeed ZeRO-3 执行，无需专用训练包装器。
单机多卡可在命令前设置 `CUDA_VISIBLE_DEVICES=0,1,2,3`。
原入口当前每卡 batch size 5；单节点梯度累积为 4，单卡有效 batch 为 20。
ZeRO-3 将优化器状态放到 CPU，训练默认使用四个 `spawn` worker 并行准备数据，避免 fork 继承大体积 CPU 优化器状态；
仍需充足的主机内存。
优化器使用 PyTorch AdamW，无需编译 DeepSpeed CPUAdam 扩展。
可加 `MAX_ITERS=2` 做短程验证，默认训练 3,000 步，`SAVE_INTERVAL` 默认 500。

### 可选：使用本目录配置的训练包装器

需要应用本目录 `config.json` 的训练覆盖参数或自定义输出路径时，仍可从仓库根目录运行：

```bash
CUDA_VISIBLE_DEVICES=0 NGPUS=1 bash alignment/humangen_pairs/run_train.sh
```

例如使用四张 GPU 并指定训练步数和独立日志目录：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NGPUS=4 \
  bash alignment/humangen_pairs/run_train.sh \
  --max_iters=1000 --logdir=/absolute/path/to/new_run
```

`run_train.sh` 使用指定的 HOST_Alignment Python，通过 torchrun 调用 `train_humangen.py`，
后者复用原有 `alignment/train.py`。它设置生成数据的相机映射并关闭 `JOINTS.USE_JOINTS`，
因为此转换只提供视觉配对。模型路径继续使用 `alignment/config.py` 的统一配置。
其他参数沿用原训练程序的 flags；自定义本目录配置可传 `--humangen_config=/path/to/config.json`。
启动器默认单节点、单 GPU，未提供 W&B key 时使用离线日志。
请在可访问 CUDA GPU 的终端启动训练；转换和 processor/加载器验证均可使用 CPU。

若更换 `config.json` 的 `output_root` 后使用原入口，需同步更新 `alignment/config.py`
中的相机映射目录；上述包装器会自动应用该设置。

## 配置文件、位置与作用

以下路径均相对于 HOST 仓库根目录。按主 README 使用原训练入口时，主要检查
`alignment/config.py`、`alignment/scripts/ds_config_zero3.json` 和
`alignment/train_scripts/run_ds.sh`。

| 文件位置 | 作用与主要参数 | 何时修改 |
|---|---|---|
| [`alignment/config.py`](../config.py) | 模型与训练超参数：`MODEL.BASE_MODEL.MODEL_PATH` 指向本地权重；`DATA.CAM_MAPPING_DIR` 指向相机映射；`JOINTS.USE_JOINTS=False` 关闭关节输入；`TRAIN.*` 控制采样帧数；`ALIGNMENT.*` 控制 TCC/DTW 损失；`DS_CONFIG.*` 设置 batch size、优化器、学习率和调度器 | 更换模型、数据映射或训练超参数 |
| [`alignment/scripts/ds_config_zero3.json`](../scripts/ds_config_zero3.json) | DeepSpeed 运行配置：bf16、ZeRO stage 3、CPU 优化器卸载、通信桶和保存时权重聚合；`zero_force_ds_cpu_optimizer=false` 配合 `config.py` 的 `torch_adam=True` 使用 PyTorch AdamW | 调整精度、显存/主机内存分配或通信策略 |
| [`alignment/train_scripts/run_ds.sh`](../train_scripts/run_ds.sh) | 原入口的启动配置：读取环境变量，计算 GPU 数与梯度累积次数，指定 DeepSpeed JSON，将参数传给 `train.py`，日志写入 `alignment/logs/host_alignment_<时间戳>/` | 通常通过环境变量调整；修改启动逻辑时才编辑文件 |
| [`alignment/humangen_pairs/config.json`](./config.json) | 数据转换配置：`source_root`、`output_root`、`datasets`、`workers`；`training.num_frames` 同时作为转换时的最短帧数门槛；`training` 中的训练参数供专用包装器使用 | 更换原始数据位置、转换输出或数据源，以及使用包装器训练时 |
| [`alignment/environment.yml`](../environment.yml) | `HOST_Alignment` 的 Conda/Python 依赖定义 | 创建或重建环境；日常调整训练参数不需要修改 |
| `Qwen3-VL-Embedding-8B/config.json`、`preprocessor_config.json`、`video_preprocessor_config.json` | 随模型提供的网络结构和图像/视频预处理配置，由模型与 processor 自动读取；目录由 `MODEL_PATH` 决定 | 更换兼容模型时随权重一起提供，常规训练无需手动编辑 |

### 原入口的有效参数与覆盖关系

**`train_scripts/run_ds.sh` 不读取 `humangen_pairs/config.json`。** 例如，该 JSON 的
`training.batch_size=1` 只在使用 `run_train.sh → train_humangen.py` 时生效；原入口
每卡 batch size 来自 `config.py` 的 `CONFIG.DS_CONFIG.train_micro_batch_size_per_gpu=5`。
`workers=64` 是数据转换并行度；训练的 DataLoader worker 数由 `alignment/config.py`
中的 `CONFIG.DATA.NUM_WORKERS` 控制。

| 要调整的内容 | 原入口的配置位置与当前默认值 |
|---|---|
| 本地权重目录 | `config.py` → `CONFIG.MODEL.BASE_MODEL.MODEL_PATH`，默认仓库根目录的 `Qwen3-VL-Embedding-8B/` |
| 相机映射与关节输入 | `config.py` → `CONFIG.DATA.CAM_MAPPING_DIR`，默认 `humangen_pairs/output/cam_mapping`；`CONFIG.JOINTS.USE_JOINTS=False` |
| 每卡 batch size、学习率 | `config.py` → `CONFIG.DS_CONFIG.train_micro_batch_size_per_gpu=5`、`CONFIG.DS_CONFIG.optimizer.params.lr=1e-5` |
| 每个训练进程的 CPU 计算线程数 | `config.py` → `CONFIG.TRAIN.CPU_NUM_THREADS=22`；影响 CPU AdamW 等主进程计算，不是 DataLoader worker 数 |
| 采样与参考帧数 | `config.py` → `CONFIG.TRAIN.NUM_FRAMES=24`、`CONFIG.TRAIN.NUM_ALIGN_FRAMES=24` |
| 训练数据并行与预取 | `config.py` → `CONFIG.DATA.NUM_WORKERS=4`、`CONFIG.DATA.PREFETCH_FACTOR=1`，每个 worker 预取一个 batch |
| 总更新步数、保存间隔 | 启动环境变量 `MAX_ITERS=3000`、`SAVE_INTERVAL=500`，单位均为实际优化器更新步数 |
| 使用哪些 GPU | `CUDA_VISIBLE_DEVICES`，原脚本据此统计每节点 GPU 数；未设置时通过 `nvidia-smi` 检测 |
| 节点与梯度累积 | `WORLD_SIZE` 在此脚本中表示节点数，默认 1；梯度累积为 `max(1, 4 // WORLD_SIZE)`；`RANK` 为节点编号，默认 0 |
| 分布式连接 | `MASTER_ADDR=127.0.0.1`、`MASTER_PORT=29500`；多节点时需设置可达的主节点地址及端口 |
| 恢复训练或加载初始权重 | `RESUME_DIR` 恢复 DeepSpeed 检查点；`PRETRAIN_WEIGHTS` 加载单文件 state dict 进行 warm start |
| 日志同步 | `WANDB_API_KEY`；未提供时默认 `WANDB_MODE=offline` |

`train.py` 先加载 `config.py`，再用启动参数覆盖对应配置；因此原入口的
`MAX_ITERS` 会同时覆盖 `CONFIG.TRAIN.MAX_ITERS` 和调度器的 `total_num_steps`。
DeepSpeed JSON 作为运行配置底稿，梯度累积次数由启动参数覆盖，随后再合并
`CONFIG.DS_CONFIG`；其中 `optimizer` 和 `scheduler` 整体替换 JSON 中同名配置。
修改学习率、batch size 或调度器时，应优先修改 `config.py`。

例如，从仓库根目录启动单卡训练并明确指定步数：

```bash
conda activate HOST_Alignment
cd alignment
CUDA_VISIBLE_DEVICES=0 MAX_ITERS=3000 SAVE_INTERVAL=500 \
  VIDEO_PATHS=humangen_pairs/output/HumanGen_video_paths.json \
  bash train_scripts/run_ds.sh
```

将 `MAX_ITERS` 改为 2 即为短程验证。原脚本通过上述环境变量接收覆盖参数；
需要使用 `--num_align_frames`、`--logdir` 等其他训练 flags 时，直接用 `torchrun`
调用 `train.py`，或使用支持透传 flags 的 HumanGen 包装器。
包装器先应用 `humangen_pairs/config.json` 的训练设置，再使用显式命令行 flags
覆盖对应值；`NGPUS` 和 `ALIGNMENT_PYTHON` 也仅供该包装器使用。

### 生成的数据索引与映射

以下文件由 `prepare.py` 生成，训练时读取；更换数据目录后应重新生成或核对路径。

| 默认位置 | 作用 |
|---|---|
| `alignment/humangen_pairs/output/HumanGen_video_paths.json` | 机器人 episode 路径列表，通过 `VIDEO_PATHS` 传入；文件名中的 `HumanGen` 用于查找同名数据集的相机映射 |
| `alignment/humangen_pairs/output/cam_mapping/HumanGen_cam_mapping.json` | pair 父目录到 `['images']` 的映射，确定两侧读取的相机视角 |
| `alignment/humangen_pairs/output/episodes/<数据源>/<编号>/{robot,human}/task_paths.json` | 当前 episode 的明确配对对象，`same` 中只包含另一侧的绝对路径 |
| 同一 episode 下的 `task_paths_eval.json` | eval 读取配对时使用；与训练配对相同，不代表独立验证集 |

模型路径和默认相机目录由 `config.py` 的文件位置解析；转换配置里的相对路径由
`humangen_pairs/config.json` 所在目录解析。原启动器会切换到 `alignment/`，
传入 `VIDEO_PATHS`、`RESUME_DIR`、`PRETRAIN_WEIGHTS` 的相对路径应以该目录为基准，
或直接使用绝对路径。

## 训练读取性能

原入口与 HumanGen 包装器均使用 `config.py` 的以下设置：

```python
CONFIG.DATA.NUM_WORKERS = 4
CONFIG.DATA.PREFETCH_FACTOR = 1
```

训练通过 `spawn` 启动独立 worker，把当前配置传入 worker，并跨 epoch 保留进程。
默认最多提前准备四个 batch，使视频读取、解码、数据增强和 processor 工作与 GPU 计算重叠。
`spawn` 不会像 fork 那样继承主进程中的模型和 CPU 优化器大内存；worker 与预取 batch
本身仍需额外内存。首批包含进程启动开销，应观察预热后的等待时间。
需要同步定位数据问题时，可把 `NUM_WORKERS` 改为 0；继续增加 worker 数前，先检查
主机内存和实际吞吐，避免过多并发读取争用存储。

加载器还会合并同一 pair 的视频读取请求，对帧索引排序、去重后解码，再恢复请求顺序；
机器人行与人类行仍分别做原有数据增强。参考视频的文件列表直接复用，训练时省去
仅供 eval 缓存使用的 main-only processor 输入。eval 的 main-only/paired 输入保留。

CPU 优化器卸载仍会在梯度累积边界执行主机计算和状态传输，因此 GPU 利用率可能周期性
下降。应区分数据等待和优化器更新时间；本次优化主要消除逐 batch 的同步取数等待。

第一轮仅优化数据读取时的单张 H100 80GB 短程对比（2026-10-16）：每卡 batch size 4、梯度累积 4、采样/参考
各 24 帧、bf16、CPU AdamW 卸载，两组各执行 8 个 micro-batch、2 次优化器更新。
当时仍使用 96 个 CPU 计算线程和全量词表投影，尚未应用下节的计算与缓存优化。

| 测量项 | 优化前 | 优化后 |
|---|---:|---:|
| 固定 4 对样本的 CPU 输入准备 | 23.9 秒 | 11.5 秒 |
| 预热后每 batch 平均取数等待 | 19.09 秒 | 0.36 毫秒 |
| 预热后平均 GPU 利用率 | 33.0% | 63.0% |
| 两次更新的训练阶段耗时 | 315.9 秒 | 172.5 秒 |
| 观测到的容器内存最大值 | 203.5 GiB | 206.0 GiB |

训练阶段计时不含模型加载和 worker 初始化；预热统计排除首个 micro-batch，GPU 利用率
每 500 毫秒采样。短程训练阶段吞吐约提升 1.83 倍，不代表完整训练或其他硬件的保证值。
后台仍需实际读取与处理数据，毫秒级数字指主进程等待已预取 batch 的时间。
两次 CPU 优化器更新仍累计约 34.6 秒，因此利用率不会始终为 100%。

固定随机种子的输入对比中，27 个训练 tensor 条目及配对元数据与原实现完全一致；
eval 的缓存命中/未命中输入也一致，并检查了运行时配置向 spawn worker 的传递。
多 worker 的随机数种子与串行加载不同，短程 loss 不用于比较训练质量。
调试与性能对比的临时日志、采样文件和曲线已清理，以上保留历史测量汇总。

### CPU 计算、词表投影与引用缓存

`CONFIG.TRAIN.CPU_NUM_THREADS=22` 控制每个训练进程的 PyTorch CPU 计算线程数，
由 `train.py` 启动时应用。当前容器虽然能看到 192 个逻辑 CPU，实际 CPU 配额约为
22 核，PyTorch 原默认使用 96 个线程。十亿参数的 CPU AdamW 独立测试中，
96 线程更新平均约 1.20 秒，22 线程约 0.47 秒。
这是当前机器上的测量；更换资源配额或使用多卡时，应按每个训练进程能分到的 CPU
资源调整该值。DataLoader worker 仍各自使用其默认的单个 PyTorch 计算线程。

`alignment/models.py` 的 embedding 前向固定传入 `logits_to_keep=1`。alignment 使用
隐藏特征，不使用语言模型的词表 logits；因此保留全部隐藏特征，同时仅计算最后一个
token 的词表输出。按 `(8, 4945, 4096)` 隐藏特征和 151,936 词表的隔离测试，
该投影测试峰值显存从约 12.7 GiB 降到 1.5 GiB；这不是整个训练过程的显存峰值。
模型结构、检查点参数名和 TCC/DTW 目标保持原样。

`alignment/algos/algorithm.py` 还将引用特征缓存限制在 eval 路径。原实现训练时
也会写入带梯度的引用特征，但训练并不读取缓存；这会保留无用的计算图并增加显存占用。
训练现在不写该缓存，eval 仍正常缓存引用特征。

第二轮真实训练对比沿用单张 H100 80GB、batch size 4、梯度累积 4、采样/参考各
24 帧、两个预取 worker；两组各执行 4 个 micro-batch、1 次实际优化器更新：

| 测量项 | 第二轮优化前 | 第二轮优化后 |
|---|---:|---:|
| CPU 优化器更新时间 | 18.96 秒 | 8.90 秒 |
| 一次更新的训练阶段耗时 | 95.59 秒 | 85.62 秒 |
| 整个训练阶段的 PyTorch 峰值显存 | 54.79 GiB | 51.51 GiB |

训练阶段计时不含模型加载和 worker 初始化，包含首批取数等待；短程吞吐约提高 12%。
固定随机种子下，更新前四个 micro-batch 的 loss 逐项完全一致，两组投影权重均实际
更新；优化后整个训练期间的引用缓存条目数为 0。该短程检查不代表完整训练收敛结果。

### 为什么 GPU 利用率仍会高低交替

当前单卡启动器设置梯度累积为 4，`scripts/ds_config_zero3.json` 设置
`zero_optimization.offload_optimizer.device="cpu"`。每次更新依次经历：

```text
4 个 micro-batch 的 GPU 前向/反向 → 梯度传输与 CPU AdamW 更新 → 下一组计算
```

GPU 计算期间可以满载；CPU 更新和传输期间会出现低谷，甚至短时为 0。
前述短程训练中，仅优化器更新就约需 8.90 秒，因此预取解决取数等待后仍会有周期性停顿。
首批 worker 初始化和定期保存大型检查点也会带来额外等待。评估优化应看稳定阶段的
每次更新耗时和吞吐，并结合分阶段计时，而不是要求每个 `nvidia-smi` 采样点都满载。

`alignment/models.py` 还将 CLS 特征的逐个标量索引改为 `index_select` 批量提取，
减少逐组读取 GPU 索引引起的同步。48 个 CLS 的单行检查中，标量读取从 51 次降到
3 次；8 行、每行 48 组的 CPU 隔离检查从 401 次降到 17 次，特征与梯度逐项完全一致，
同时覆盖多行主/参考顺序、额外 CLS 截断和缺失 CLS 补零。该检查没有加载完整模型，
**不代表整段 GPU 训练的提速倍数**。修改需在下次启动训练时生效。

进一步缩短长低谷，应优先测量 CPU 优化器。当前使用
`config.py` 的 `CONFIG.DS_CONFIG.optimizer.params.torch_adam=True`；切换 DeepSpeed
CPUAdam 前需先解决编译环境：本机 `nvcc` 为 12.9，PyTorch 的 CUDA 构建版本为
12.4，当前安装的 DeepSpeed 构建器拒绝该组合。完成兼容编译及实际参数更新、保存/恢复
验证后再比较吞吐。单张 80GB 卡上直接关闭 CPU offload 可能超出显存；使用更多显存或
多卡分片后再评估将优化器状态放回 GPU。增加 worker 数主要针对取数等待，无法消除
CPU 更新阶段的停顿。

### 使用 CPU optimizer offload 的原因

2026-10-16 在独占的单张 H100 80GB 上关闭 optimizer offload，实际执行原 `train.py`，保持每卡 batch 4、
梯度累积 4、采样/参考各 24 帧。DeepSpeed 初始化成功，但**第一个前向在视觉模块
旋转位置编码处 OOM，实际参数更新次数为 0**。PyTorch 峰值已分配显存约 76.82 GiB，
`nvidia-smi` 采样峰值约 79.05 GiB；报错时剩余 138.25 MiB，下一次分配需要 608 MiB。
因此关闭卸载的设置在当前单卡及训练设置下无法运行，也没有可比较的更新吞吐。

实际模型含 8,767,648,112 个参数。按当前 ZeRO-3/PyTorch AdamW 实现，单卡的 FP32
主权重约 32.66 GiB，两份 Adam 动量约 65.32 GiB，合计约 97.99 GiB，另外还需 bf16
权重、梯度、激活和缓冲区。这是容量估算；Adam 动量在首次更新时延迟创建，不能将
估算值当作此次故障时已分配的显存。降低 batch 或帧数主要减少激活，无法单独解决
全部优化器状态留在当前单卡上的容量问题。

训练继续使用 `scripts/ds_config_zero3.json` 中的 CPU offload 配置。
关闭 offload 的试验配置与临时运行文件已清理。

同日测试 ZeRO-2 + CPU offload 时，4 个 micro-batch 的前向/反向均完成，但首次更新时
容器内存达到 222.59 GiB，触发测试保护，完成更新数为 0。该试验是主动内存保护终止，
未等待系统 OOM，也未发生 CUDA OOM；当前约 230 GiB 的内存资源下继续使用 ZeRO-3。
其余调优配置、试验入口和运行产物已清理。

## 输出布局与边界

```text
output/
├── HumanGen_video_paths.json       # 每对只列 robot，训练方向 robot → human
├── cam_mapping/HumanGen_cam_mapping.json
├── pairs.jsonl                     # 原始身份、源视频、新路径、两侧帧数
├── skipped.jsonl                   # 跳过原因与来源
├── summary.json
├── validation_report.json         # validate.py 生成
├── compatible_videos/             # 仅不兼容的 AV1 视频需要 H.264 副本
└── episodes/{dataset}/{manifest_position}/
    ├── robot/
    │   ├── images.mp4 -> 原机器人视频
    │   ├── instruction.txt
    │   ├── task_paths.json         # {"same": [human episode 的绝对路径]}
    │   └── task_paths_eval.json
    └── human/                     # 相同结构，same 指回 robot
```

相机映射以每个 pair 的父目录为 key；两侧 `same` 只包含对方，因此不会混入机器人–机器人
配对或退回 self-alignment。两侧视频帧数可以不同，时序对应由 alignment 模型学习。
`task_paths_eval.json` 供读取验证使用，**不是独立验证集**；正式报告指标前需另行按源视频/任务划分。
输出用于视觉 alignment 训练，不包含策略训练要求的动作、关节或文本 embedding 文件。
全量转换检查视频首帧和帧数；抽样验证解码实际训练采样帧，不等同于逐帧完整审计或模型训练验证。

配对身份、重新编号、帧区间歧义和 RoboTwin 路径规则的回归检查：

```bash
/data/250010175/miniforge3/envs/HOST_Alignment/bin/python \
  alignment/humangen_pairs/test_prepare.py
```
