# Stage Advantage Step 1 训练踩坑记录与解决方案

> 本文档记录了使用少量数据（`test_data-525_lerobot`，3 episodes / 1844 帧）跑通 `stage_advantage` Step 1（Train Advantage Estimator）过程中遇到的所有问题及解决方案。

---

## 环境信息

- **项目**：kai0（基于 openpi）
- **训练脚本**：`scripts/train_pytorch.py`
- **Config 名称**：`ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD`
- **数据**：`test_data-525_lerobot`（LeRobot v2.1 格式）
- **机器**：单 GPU

---

## 问题 1：缺少 PyTorch 格式的 pi0.5 base checkpoint

### 现象

`scripts/train_pytorch.py` 需要 `pytorch_weight_path` 指向一个包含 `model.safetensors` 的 PyTorch checkpoint，但项目中没有。

### 原因

- openpi 官方在 GCS (`gs://openpi-assets/checkpoints/pi05_base`) 只提供了 **JAX (Orbax)** 格式的 base checkpoint
- Physical Intelligence 的 HuggingFace 上也没有发布 PyTorch 版本的 pi0.5 权重
- kai0 的 `scripts/download_checkpoints.py` 只下载 fine-tuned 的任务模型，不下载 base checkpoint

### 解决方案

openpi 官方提供了 JAX → PyTorch 转换脚本，从 GitHub 获取并执行转换：

```bash
# 1. 下载转换脚本
curl -sL "https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/examples/convert_jax_model_to_pytorch.py" \
  -o scripts/convert_jax_model_to_pytorch.py

# 2. 下载 JAX checkpoint（会自动缓存到 ~/.cache/openpi/）
uv run python -c "
from openpi.shared.download import maybe_download
path = maybe_download('gs://openpi-assets/checkpoints/pi05_base')
print(path)
"

# 3. 但实测下来上述办法下载十分缓慢，于是采用下述办法
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base ./checkpoints/.cache/

# 4. 执行转换（约 5-10 分钟）
uv run python scripts/convert_jax_model_to_pytorch.py \
    --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_base \
    --config_name pi05_libero \
    --output_path ./checkpoints/pi05_base

# 5. 手动复制 assets（转换脚本不会自动复制）
cp -r ~/.cache/openpi/openpi-assets/checkpoints/pi05_base/assets ./checkpoints/pi05_base/
```

转换完成后，修改 `src/openpi/training/config.py`：

```python
pytorch_weight_path="/mnt/pfs/zhangjiyao/yiming/kai0/checkpoints/pi05_base",
```

---

## 问题 2：HuggingFace SSL 连接错误（`SSLEOFError`）

### 现象

使用代理访问 HuggingFace Hub 时，Python requests 库抛出 SSL 错误：

```
requests.exceptions.SSLError: (MaxRetryError(
    "HTTPSConnectionPool(host='huggingface.co', port=443): ...
    Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] ...'))"
))
```

### 原因

当前环境的 HTTP 代理与 Python requests 的 HTTPS 请求存在兼容性问题，导致 SSL 握手失败。

### 解决方案

设置 HuggingFace 环境变量，**禁用 SSL 验证**：

```bash
export HF_HUB_DISABLE_SSL_VERIFICATION=1
```

> ⚠️ 这只是为了在当前网络环境下绕过代理的 SSL 问题。如果网络环境正常，不需要设置。

---

## 问题 3：LeRobot 无法直接加载本地数据集

### 现象

`repo_id` 指向本地绝对路径时，`LeRobotDatasetMetadata` 把路径当作 HuggingFace Hub 的 repo ID 去验证，报错：

```
HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name':
'/mnt/pfs/.../test_data-525_lerobot'
```

### 原因

LeRobot v2.1 的 `LeRobotDatasetMetadata` 在初始化时会强制连接 HuggingFace Hub 验证版本，即使传入 `root` 参数也无法完全绕过。

### 解决方案

**把数据集上传到 HuggingFace Hub，通过 Hub 加载。**

步骤：

```bash
# 1. 设置 HF token 和代理
export HF_TOKEN="your_token"
export http_proxy=http://10.66.65.186:18000
export https_proxy=http://10.66.65.186:18000
export HF_HUB_DISABLE_SSL_VERIFICATION=1

# 2. 创建 dataset repo
huggingface-cli repo create your-username/test-data-525-lerobot --repo-type dataset

# 3. 上传本地数据集
huggingface-cli upload your-username/test-data-525-lerobot \
    /path/to/test_data-525_lerobot \
    --repo-type dataset
```

上传后修改 `config.py`：

```python
repo_id = "your-username/test-data-525-lerobot",
```

---

## 问题 4：数据集缺少 LeRobot v2.1 必需的 version tag

### 现象

数据集上传后，LeRobot 加载时报错：

```
RevisionNotFoundError: Your dataset must be tagged with a codebase version.
```

### 原因

LeRobot v2.1 要求数据集在 HuggingFace 上有一个 version tag（如 `v2.1`），对应 `info.json` 中的 `codebase_version`。

### 解决方案

给 HuggingFace repo 打上 version tag：

```bash
export HF_HUB_DISABLE_SSL_VERIFICATION=1
export HF_TOKEN="your_token"

python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_tag('your-username/test-data-525-lerobot', tag='v2.1', repo_type='dataset')
print('Tag created')
"
```

---

## 问题 5：数据集缺少 `meta/episodes_stats.jsonl`

### 现象

```
FileNotFoundError: [Errno 2] No such file or directory:
'.../meta/episodes_stats.jsonl'
```

### 原因

LeRobot v2.1 数据集要求每个 episode 的统计信息（mean、std、min、max 等）存储在 `meta/episodes_stats.jsonl` 中。原始数据创建时未生成此文件。

### 解决方案

用脚本本地生成并重新上传：

```bash
# 在项目根目录执行
uv run python generate_episodes_stats.py
```

脚本内容（已保存在 `generate_episodes_stats.py`）：

- 读取每个 episode 的 parquet
- 计算数值型 feature 的 count、mean、std、min、max
- 写入 `meta/episodes_stats.jsonl`

生成后上传：

```bash
huggingface-cli upload your-username/test-data-525-lerobot \
    ./test_data-525_lerobot/meta/episodes_stats.jsonl \
    meta/episodes_stats.jsonl \
    --repo-type dataset
```

---

## 问题 6：数据集 chunk 编号不匹配

### 现象

```
FileNotFoundError: Unable to find '.../data/chunk-001/episode_000004.parquet'
```

### 原因

`info.json` 中 `chunks_size=3`，LeRobot 按 `episode_index // chunks_size` 计算 chunk 编号：

- episode 4 → `4 // 3 = 1` → 期望在 `chunk-001/`
- 但文件实际在 `chunk-000/`

### 解决方案

增大 `chunks_size`，让所有 episode 都落在 `chunk-000` 中：

```bash
# 修改 info.json
python3 -c "
import json
with open('test_data-525_lerobot/meta/info.json') as f:
    info = json.load(f)
info['chunks_size'] = 10  # 大于总 episode 数即可
with open('test_data-525_lerobot/meta/info.json', 'w') as f:
    json.dump(info, f, indent=2)
"
```

修改后重新上传 `info.json` 到 HuggingFace。

---

## 问题 7：`KeyError: 'progress_gt'`

### 现象

训练启动后，在 `transforms.py` 的 `RepackTransform` 中报错：

```
KeyError: 'progress_gt'
```

### 原因

`config.py` 的 `repack_transforms` 中配置了 `"progress_gt": "progress_gt"`，但数据集中只有 `stage_progress_gt` 列，没有 `progress_gt` 列。

### 解决方案

修改 `src/openpi/training/config.py`，把 `progress_gt` 映射到 `stage_progress_gt`：

```python
# 修改前
"progress_gt": "progress_gt",
"stage_progress_gt": "stage_progress_gt",

# 修改后
"progress_gt": "stage_progress_gt",
"stage_progress_gt": "stage_progress_gt",
```



---

## 问题 8：训练配置参数不适合小数据量 / 单 GPU

### 现象

- `batch_size=256` → 单 GPU OOM
- `num_workers=8` → 进程数过多
- `num_train_steps=50_000` → 对于 1844 帧的 smoke test 需要几十小时

### 解决方案

修改 `config.py` 中 `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD` 的参数：

```python
num_train_steps=1_000,      # smoke test 用 1000 步即可
batch_size=8,               # 单 GPU 小显存
num_workers=2,              # 减少进程数
save_interval=500,          # 保存频率
```

> 正式训练时需要调回 `num_train_steps=50_000` 和 `batch_size=256`（多 GPU）。

---

## 完整启动命令

```bash
cd /mnt/pfs/zhangjiyao/yiming/kai0

# 1. 设置必要的环境变量
export WANDB_API_KEY="your_wandb_key"
export http_proxy=http://10.66.65.186:18000
export https_proxy=http://10.66.65.186:18000
export HF_HUB_DISABLE_SSL_VERIFICATION=1

# 2. 启动训练
uv run python scripts/train_pytorch.py \
    ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD \
    --exp_name=smoke_test
```

---

## 依赖检查清单

在运行前确保：

- [ ] `checkpoints/pi05_base/` 下有 `model.safetensors`、`config.json`、`assets/`
- [ ] `config.py` 中 `repo_id` 指向 HuggingFace 数据集（非本地路径）
- [ ] `config.py` 中 `pytorch_weight_path` 指向正确的 pi05_base 路径
- [ ] HuggingFace 数据集已上传 `meta/episodes_stats.jsonl`
- [ ] HuggingFace 数据集已打 `v2.1` tag
- [ ] `info.json` 中 `chunks_size` 足够大
- [ ] `config.py` 中 `progress_gt` 已映射到 `stage_progress_gt`
- [ ] `batch_size`、`num_workers`、`num_train_steps` 已根据硬件调整

---

## 训练成功标志

正常启动后，控制台会显示类似输出：

```
Training:   1%| | 10/1000 [00:25<42:15, 2.57s/it, loss=0.0025, lr=2.50e-05, step=10]
```

- `loss` 在 0.001 ~ 0.01 之间波动是正常的
- wandb 会自动记录训练曲线

---

*文档生成时间：2026-05-27*
