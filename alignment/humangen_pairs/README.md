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

### 3. 启动训练：`run_train.sh`

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

不要仅把生成的路径列表传给原 `train_scripts/run_ds.sh`：原配置仍指向内部相机映射并开启关节输入。

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
