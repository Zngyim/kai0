# Stage Advantage Step 0 & Step 1 完整指南

> 本文档汇总了从原始数据准备到 Advantage Estimator 训练完成的完整流程，涵盖所有关键脚本的使用方法。

---

## 流程概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 0: 数据准备 & stage_progress_gt 标注                                   │
│  ├── 原始数据 → LeRobot v2.1 格式 (convert_testdata_to_lerobot.py)          │
│  ├── 或：为已有 LeRobot 数据集补充 stage_progress_gt (annotate_*.py)        │
│  ├── 生成 meta/episodes_stats.jsonl (generate_episodes_stats.py)            │
│  └── 上传至 HuggingFace Hub (upload_lerobot_to_hf.py)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Step 1: 训练 Advantage Estimator (scripts/train_pytorch.py)                │
│  ├── JAX → PyTorch base checkpoint 转换                                      │
│  ├── 修改 config.py                                                          │
│  └── 启动训练                                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 0: 数据准备与 stage_progress_gt 标注

### 0.1 前置要求

所有脚本均需在项目根目录执行，且虚拟环境已激活：

```bash
cd /mnt/pfs/zhangjiyao/yiming/kai0
source .venv/bin/activate
# 或：uv run <command>
```

LeRobot v2.1 数据集的标准目录结构：

```
my_dataset/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       ├── observation.images.top_head/
│       │   ├── episode_000000.mp4
│       │   └── ...
│       └── ...
└── meta/
    ├── info.json
    ├── episodes.jsonl
    ├── episodes_stats.jsonl   # <-- 需要生成
    └── tasks.jsonl
```

### 0.2 场景 A：从 HDF5 + debug_video 原始数据开始

如果你的原始数据是 `test_data-525` 这种格式（包含 `data/episode*.hdf5` + `debug_video/`），使用 `scripts/convert_testdata_to_lerobot.py` 一次性转换为 LeRobot v2.1 格式。

**输入数据要求：**

```
test_data-525/
├── data/
│   ├── episode_0.hdf5
│   ├── episode_1.hdf5
│   └── ...
├── debug_video/
│   ├── top_head/
│   │   ├── episode_0/
│   │   │   ├── 0.jpg
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── scene_info.json
└── augmentation_metadata.json   # <-- stage_progress_gt 的计算依据
```

其中 `augmentation_metadata.json` 的格式如下（每个 episode 一个 key）：

```json
{
  "0": {
    "subtask_completion_indices": [230, 371, 575],
    "segments": [
      {
        "perturb_start_hdf5": 22,
        "perturb_end_hdf5": 41,
        "recovery_start_hdf5": 41,
        "recovery_end_hdf5": 58
      }
    ]
  }
}
```

**执行转换：**

```bash
uv run python scripts/convert_testdata_to_lerobot.py \
    --input-dir ./test_data-525 \
    --output-dir ./test_data-525_lerobot \
    --k-stages 3
```

转换脚本会自动：
- 读取 HDF5 中的 `observation.state`、`action` 等数据
- 读取 `augmentation_metadata.json` 计算 `stage_progress_gt`
- 将视频帧编码为 H.264 MP4
- 生成 `meta/info.json`、`meta/episodes.jsonl`、`meta/tasks.jsonl`

### 0.3 场景 B：为已有 LeRobot 数据集补充 stage_progress_gt

如果你已经有一个 LeRobot v2.1 数据集，但缺少 `stage_progress_gt` 列，使用 `scripts/annotate_stage_progress_gt.py`：

```bash
uv run python scripts/annotate_stage_progress_gt.py \
    --dataset-dir ./my_lerobot_dataset \
    --aug-meta ./augmentation_metadata.json \
    --k-stages 3
```

该脚本会：
- 遍历 `data/chunk-*/episode_*.parquet`
- 根据 `augmentation_metadata.json` 计算 `stage_progress_gt`
- 直接更新 parquet 文件（原地修改，建议先备份）

**先 dry-run 确认无误：**

```bash
uv run python scripts/annotate_stage_progress_gt.py \
    --dataset-dir ./my_lerobot_dataset \
    --aug-meta ./augmentation_metadata.json \
    --k-stages 3 \
    --dry-run
```

### 0.4 生成 episodes_stats.jsonl

LeRobot v2.1 要求每个 episode 的统计信息存储在 `meta/episodes_stats.jsonl` 中。使用 `generate_episodes_stats.py`：

```bash
uv run python generate_episodes_stats.py
```

> 注意：该脚本默认硬编码了 `test_data-525_lerobot` 路径。如果用于其他数据集，请先修改脚本中的 `dataset_root` 变量。

### 0.5 上传数据集到 HuggingFace Hub

LeRobot 的 `LeRobotDatasetMetadata` 在加载时会强制连接 HuggingFace Hub 验证，因此**推荐将数据集上传到 Hub**。

使用一体化脚本 `scripts/upload_lerobot_to_hf.py`（同时解决上传、打 tag、生成 stats）：

```bash
export HF_TOKEN="your_hf_token"

# 一次性完成：生成 stats → 上传 → 打 version tag
uv run python scripts/upload_lerobot_to_hf.py \
    --dataset-dir ./test_data-525_lerobot \
    --repo-id your-username/test-data-525-lerobot \
    --private
```

脚本会自动：
1. 读取 `meta/episodes.jsonl`，遍历每个 episode 的 parquet 生成 `meta/episodes_stats.jsonl`
2. 将整个数据集上传到 HuggingFace Hub
3. 读取 `meta/info.json` 中的 `codebase_version`，自动打对应 tag（如 `v2.1`）

如果只想上传、跳过 stats 生成：

```bash
uv run python scripts/upload_lerobot_to_hf.py \
    --dataset-dir ./test_data-525_lerobot \
    --repo-id your-username/test-data-525-lerobot \
    --skip-stats
```

### 0.6 Step 0 检查清单

- [ ] `data/chunk-*/episode_*.parquet` 中包含 `stage_progress_gt` 列
- [ ] `meta/info.json` 中 `codebase_version` 已设置为 `"v2.1"`
- [ ] `meta/episodes_stats.jsonl` 已生成
- [ ] HuggingFace repo 已上传且包含 `v2.1` tag

---

## Step 1: 训练 Advantage Estimator

### 1.1 准备 PyTorch 格式的 Base Checkpoint

`scripts/train_pytorch.py` 需要 PyTorch 格式的 pi0.5 base checkpoint，但 openpi 官方只提供 JAX (Orbax) 格式。需要先转换：

```bash
# 1. 下载转换脚本（如尚未下载）
curl -sL "https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/examples/convert_jax_model_to_pytorch.py" \
  -o scripts/convert_jax_model_to_pytorch.py

# 2. 下载 JAX checkpoint（自动缓存到 ~/.cache/openpi/）
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base ./checkpoints/.cache/

# 3. 执行转换（约 5-10 分钟）
uv run python scripts/convert_jax_model_to_pytorch.py \
    --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_base \
    --config_name pi05_libero \
    --output_path ./checkpoints/pi05_base

# 4. 复制 assets（转换脚本不会自动复制）
cp -r ~/.cache/openpi/openpi-assets/checkpoints/pi05_base/assets ./checkpoints/pi05_base/
```

转换完成后，`checkpoints/pi05_base/` 下应包含：
- `model.safetensors`
- `config.json`
- `assets/`

### 1.2 修改训练配置

编辑 `src/openpi/training/config.py` 中的 `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD`（或你使用的配置）：

```python
TrainConfig(
    name="ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD",
    ...
    data=LerobotAgilexDataConfig(
        repo_id="your-username/test-data-525-lerobot",  # <-- 修改为你的 HF repo
        ...
    ),
    pytorch_weight_path="/mnt/pfs/zhangjiyao/yiming/kai0/checkpoints/pi05_base",  # <-- 确认路径
    ...
)
```

**关键配置项说明：**

| 配置项 | 说明 |
|--------|------|
| `repo_id` | HuggingFace dataset repo ID（或本地绝对路径，但推荐 HF Hub） |
| `pytorch_weight_path` | PyTorch 格式 pi0.5 base checkpoint 路径 |
| `num_train_steps` | 总训练步数（smoke test 用 1,000；正式训练 50,000） |
| `batch_size` | 单 GPU 用 8，8 GPU 用 256 |
| `num_workers` | DataLoader worker 数（单 GPU 推荐 2） |
| `save_interval` | 保存 checkpoint 的间隔步数 |
| `skip_norm_stats=True` | Advantage estimator 不需要 norm stats |

### 1.3 启动训练

```bash
export WANDB_MODE=${WANDB_MODE:-offline}
export HF_HUB_DISABLE_SSL_VERIFICATION=1  # 如遇到 SSL 问题

# 单 GPU
uv run python scripts/train_pytorch.py \
    ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD \
    --exp_name=run1 \
    --save_interval 10000

# 多 GPU DDP（8 卡）
uv run torchrun --standalone --nproc_per_node=2 \
    scripts/train_pytorch.py \
    ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD \
    --exp_name=run1 \
    --save_interval 10000
```

### 1.4 训练输出

Checkpoint 保存在：

```
experiment/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/
└── run1/
      ├── 10000/
      │     ├── model.safetensors
      │     ├── optimizer.pt
      │     └── metadata.pt
      ├── 20000/
      └── ...
```

### 1.5 Step 1 检查清单

- [ ] `checkpoints/pi05_base/` 下包含 `model.safetensors`、`config.json`、`assets/`
- [ ] `config.py` 中 `repo_id` 指向正确的 HuggingFace 数据集
- [ ] `config.py` 中 `pytorch_weight_path` 指向正确的 base checkpoint
- [ ] `config.py` 中 `progress_gt` 已映射到 `stage_progress_gt`
- [ ] `batch_size`、`num_workers`、`num_train_steps` 已根据硬件调整

---

## 常见问题速查

详细排错请参考项目根目录的 `STAGE_ADVANTAGE_STEP1_TROUBLESHOOTING.md`。

| 问题 | 快速解决 |
|------|---------|
| `SSLEOFError` 连接 HF Hub | `export HF_HUB_DISABLE_SSL_VERIFICATION=1` |
| `HFValidationError` 本地路径无法加载 | 上传到 HuggingFace Hub（见 0.5） |
| `RevisionNotFoundError: must be tagged` | 使用 `upload_lerobot_to_hf.py` 自动打 tag |
| `FileNotFoundError: episodes_stats.jsonl` | 使用 `upload_lerobot_to_hf.py` 自动生成 |
| `KeyError: 'progress_gt'` | 修改 `config.py` 的 RepackTransform：`"progress_gt": "stage_progress_gt"` |
| 单 GPU OOM | `batch_size=8`, `num_workers=2` |
| chunk 编号不匹配 | 修改 `info.json` 中 `chunks_size` > 总 episode 数 |

---

## 附录：脚本位置汇总

| 脚本 | 路径 | 用途 |
|------|------|------|
| `convert_testdata_to_lerobot.py` | `scripts/convert_testdata_to_lerobot.py` | HDF5 + video → LeRobot v2.1 |
| `annotate_stage_progress_gt.py` | `scripts/annotate_stage_progress_gt.py` | 为已有 LeRobot 数据集补充 stage_progress_gt |
| `generate_episodes_stats.py` | `generate_episodes_stats.py` | 生成 `meta/episodes_stats.jsonl` |
| `upload_lerobot_to_hf.py` | `scripts/upload_lerobot_to_hf.py` | 生成 stats + 上传 HF + 打 tag |
| `train_pytorch.py` | `scripts/train_pytorch.py` | Step 1 训练 Advantage Estimator |
| `convert_jax_model_to_pytorch.py` | `scripts/convert_jax_model_to_pytorch.py` | JAX → PyTorch checkpoint 转换 |

---

*文档生成时间：2026-05-27*
