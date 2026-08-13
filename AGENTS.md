# AGENTS.md — χ₀ (kai0)

特别重要：在本项目中必须使用中文与用户对话。

This file contains project-specific guidance for AI coding agents working on the χ₀ (kai0) codebase. The reader is assumed to know nothing about the project.

---

## Project Overview

χ₀ (pronounced "kai0") is a resource-efficient framework for achieving production-level robustness in robotic manipulation by taming distributional inconsistencies. It is built on top of [openpi](https://github.com/Physical-Intelligence/openpi) — the open-source release of Physical Intelligence's π₀ and π₀.5 vision-language-action models.

The project addresses systematic distributional shift among the human demonstration distribution, the policy's inductive bias, and the test-time execution distribution through three technical modules:

- **Model Arithmetic** (`model_arithmetic/`): Weight-space merging of multiple trained checkpoints to aggregate knowledge without Mixture-of-Experts architectures.
- **Stage Advantage** (`stage_advantage/`): Stage-aware advantage estimation that provides stable, dense progress signals for policy training via Advantage-Weighted Behavior Cloning (AWBC).
- **Train-Deploy Alignment** (`train_deploy_alignment/`): Bridges the train-deploy gap via spatio-temporal data augmentation, DAgger-style policy-in-the-loop collection, and temporal chunk-wise smoothing/ensembling at inference.

The codebase supports dual-arm garment manipulation tasks (flattening/folding, hanging, sorting) on Agilex Piper and ARX X5 robots.

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python ≥3.11 |
| Package Manager | [uv](https://docs.astral.sh/uv/) (lockfile: `uv.lock`) |
| Main ML Framework | JAX 0.5.3 (CUDA 12) + Flax 0.10.2 + NNX |
| Secondary ML Framework | PyTorch 2.7.1 (used only for advantage estimator training) |
| Checkpointing | Orbax 0.11.13 (JAX), `safetensors` (PyTorch) |
| Dataset Format | [LeRobot](https://github.com/huggingface/lerobot) v2.1 (parquet + video chunks + meta) |
| Logging | Weights & Biases (`wandb`) |
| CLI Parsing | `tyro` (used heavily for dataclass-based CLIs) |
| Linting / Formatting | Ruff 0.8.6 (pre-commit hook) |
| Build Backend | Hatchling |

Key dependencies are pinned tightly (e.g., `jax[cuda12]==0.5.3`, `flax==0.10.2`, `transformers==4.53.2`) because the training pipeline is sensitive to exact versions.

---

## Build and Environment Setup

```bash
# 1. Clone with submodules
git clone --recurse-submodules git@github.com:OpenDriveLab/kai0.git
# Or if already cloned:
git submodule update --init --recursive

# 2. Sync dependencies with uv
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 3. (Optional) Install pre-commit hooks
pre-commit install
```

- **Python version**: Locked to 3.11 (see `.python-version`).
- **Virtual environment**: Managed automatically by `uv` (`.venv/`).
- **Workspace**: The root `pyproject.toml` declares `packages/*` as workspace members. `packages/openpi-client/` is a sub-package with its own `pyproject.toml`.

### Docker (Optional)

Docker files live in `scripts/docker/`:
- `serve_policy.Dockerfile` — for serving the policy server
- `compose.yml` — Docker Compose setup
- Helper scripts for installing Docker and NVIDIA Container Toolkit on Ubuntu 22.04

> Note: Docker has not been thoroughly tested in the kai0 project yet.

---

## Code Organization

```
├── src/openpi/                    # Main Python package
│   ├── models/                    # JAX model definitions
│   │   ├── pi0.py                 # π₀ architecture
│   │   ├── pi0_fast.py            # π₀-FAST variant
│   │   ├── pi0_config.py          # Model configs (Pi0Config, Pi0FASTConfig, etc.)
│   │   ├── pi0_rtc.py             # Real-time chunking variant
│   │   ├── gemma.py / gemma_fast.py / siglip.py / vit.py  # Backbones
│   │   ├── tokenizer.py           # Paligemma / FAST tokenizers
│   │   ├── lora.py                # LoRA support
│   │   └── model.py               # Base model abstraction & action generation
│   ├── models_pytorch/            # PyTorch model implementations
│   │   ├── pi0_pytorch.py         # PyTorch π₀ + AdvantageEstimator
│   │   ├── gemma_pytorch.py       # PyTorch Gemma
│   │   ├── preprocessing_pytorch.py
│   │   └── transformers_replace/  # Patched transformers modules (PaliGemma, SigLIP, Gemma)
│   ├── policies/                  # Robot-environment-specific policy configs
│   │   ├── agilex_policy.py       # Agilex Piper (Task_A, Task_B)
│   │   ├── arx_policy.py          # ARX X5 (Task_C)
│   │   ├── aloha_policy.py        # Aloha / Trossen
│   │   ├── droid_policy.py        # DROID
│   │   ├── libero_policy.py       # LIBERO sim
│   │   ├── policy.py              # Generic policy interface
│   │   └── policy_config.py       # Policy construction from checkpoints
│   ├── training/                  # Training infrastructure
│   │   ├── config.py              # Central config registry (_CONFIGS list)
│   │   ├── data_loader.py         # LeRobot / RLDS data loading
│   │   ├── advantage_dataset.py   # Dataset for advantage estimator
│   │   ├── optimizer.py           # LR schedules & optimizers
│   │   ├── checkpoints.py         # Checkpoint save / restore helpers
│   │   ├── weight_loaders.py      # Loading pretrained weights into models
│   │   ├── sharding.py            # FSDP / data-parallel sharding
│   │   └── utils.py               # Training utilities
│   ├── shared/                    # Shared utilities
│   │   ├── normalize.py           # RunningStats, norm_stats save/load
│   │   ├── image_tools.py         # Image resizing, padding, uint8 conversion
│   │   ├── array_typing.py        # JAX array type annotations
│   │   ├── download.py            # GCS / HTTP download helpers
│   │   └── nnx_utils.py           # Flax NNX helpers
│   ├── transforms.py              # Data transforms (resize, tokenize, pad, repack)
│   ├── serving/                   # Inference server
│   │   └── websocket_policy_server.py
│   └── conftest.py                # pytest fixture: sets JAX to CPU if no GPU
│
├── scripts/                       # Executable scripts
│   ├── train.py                   # Main JAX training loop
│   ├── train_pytorch.py           # PyTorch training (advantage estimator)
│   ├── serve_policy.py            # Start WebSocket policy server
│   ├── compute_norm_states_fast.py# Fast norm-stats computation from local parquet
│   ├── compute_norm_stats.py      # Original openpi norm-stats script
│   ├── download_dataset.py        # Download Kai0 dataset from HuggingFace
│   ├── download_checkpoints.py    # Download best-model checkpoints
│   ├── merge_lerobot.py / split_lerobot.py  # Dataset utilities
│   └── train_test.py              # Smoke test for training
│
├── model_arithmetic/              # Checkpoint merging module
│   ├── arithmetic.py              # JAX checkpoint mixing (Orbax)
│   ├── arithmetic_torch.py        # PyTorch checkpoint mixing (safetensors)
│   ├── common.py                  # Shared helpers
│   ├── dump_data.py               # Dump validation set for weight optimization
│   └── split_data.py              # Split LeRobot dataset by episode
│
├── stage_advantage/               # Advantage estimation & AWBC module
│   ├── annotation/
│   │   ├── eval.py                # Run advantage estimator on dataset
│   │   ├── evaluator.py           # Batched GPU inference wrapper
│   │   ├── discretize_advantage.py# Bin advantages → task_index
│   │   └── discretize_advantage.sh# Batch wrapper
│   └── awbc/                      # AWBC-specific docs
│
├── train_deploy_alignment/        # Train-deploy alignment module
│   ├── data_augment/              # Time scaling, space mirroring, HDF5→LeRobot
│   ├── dagger/                    # DAgger data collection (Agilex + ARX)
│   └── inference/                 # Deployment code (temporal smoothing, RTC, ensembling)
│
├── packages/openpi-client/        # Lightweight inference client
│   └── src/openpi_client/         # WebSocket client, image_tools, runtime abstractions
│
├── setup/                         # 3D-printed gripper/camera mount CAD files
├── checkpoints/                   # Downloaded / trained checkpoints
├── data/                          # Downloaded LeRobot datasets
└── docs/                          # Additional documentation
```

---

## Configuration System

All training runs are driven by named configs in **`src/openpi/training/config.py`**. The file contains a global list `_CONFIGS` (line ~764) that registers every `TrainConfig`.

A `TrainConfig` is a frozen dataclass specifying:
- `name` — unique config identifier (used on CLI)
- `model` — e.g., `Pi0Config(pi05=True)`
- `data` — e.g., `LerobotAgilexDataConfig(repo_id=...)`
- `weight_loader` — how to load pretrained weights
- `pytorch_weight_path` — PyTorch checkpoint path (for advantage estimator)
- Training hyperparameters: `batch_size`, `num_train_steps`, `lr_schedule`, `optimizer`, `ema_decay`, etc.
- `checkpoint_base_dir` — defaults to `/mnt/bos/shared-dataset/zhangjiyao/zimu/checkpoints` (change for your setup)

**Important**: Before training, you **must** edit the configs around lines 1173–1226 (and the advantage-estimator configs around line 1222) to set absolute paths for:
- `repo_id` — path to your local dataset subset (e.g., `<repo_root>/data/FlattenFold/base`)
- `weight_loader` / `pytorch_weight_path` — path to base checkpoint

Configs are selected on the CLI by name:
```bash
uv run scripts/train.py pi05_flatten_fold_normal --exp_name=run1
```

Robot-specific data-config factories (`LerobotAgilexDataConfig`, `LerobotARXDataConfig`, `LeRobotAlohaDataConfig`, etc.) live in the same file and handle camera names, action dimensions, delta-joint flags, and repack transforms.

---

## Training Workflows

### 1. Normal π₀.5 Full Fine-Tuning

```bash
# Step 1: compute normalization stats (fast path, reads local parquet directly)
uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_normal

# Step 2: train
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_normal --exp_name=run1
```

Checkpoints are saved under `<checkpoint_base_dir>/<config_name>/<exp_name>/`.

### 2. Advantage Estimator (PyTorch)

```bash
# Single GPU
uv run python scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD --exp_name=run1 --save_interval 10000

# Multi-GPU DDP
uv run torchrun --standalone --nproc_per_node=8 scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD --exp_name=run1 --save_interval 10000
```

Outputs go to `experiment/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/<exp_name>/`.

### 3. AWBC Training (after Stage Advantage pipeline)

```bash
uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_awbc
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_awbc --exp_name=run1
```

### 4. Model Arithmetic (Checkpoint Merging)

```bash
# Dump validation data
python model_arithmetic/dump_data.py --dataset pi05_hang_cloth --output val.pkl

# Mix JAX checkpoints
python model_arithmetic/arithmetic.py \
  --config pi05_hang_cloth --data-path val.pkl \
  --checkpoints /path/to/ckpt1/90000 /path/to/ckpt2/90000 \
  --output /path/to/mixed --optimize_method inverse_loss --use_gpu --gpu_ids "0"
```

Supported methods: `average`, `inverse_loss`, `gradient_descent`, `adaptive_gradient_descent`, `greedy`, or manual `--weights`.

---

## Inference and Deployment

### Policy Server

Start a WebSocket policy server on the GPU host:

```bash
# Default env policies
uv run scripts/serve_policy.py --env=ALOHA_SIM

# Custom checkpoint
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_flatten_fold_normal \
  --policy.dir=/path/to/checkpoint/90000 \
  --port=8000
```

### Robot Client

The robot-side machine runs a minimal client. Install the client package:
```bash
cd packages/openpi-client && pip install -e .
```

Connect and infer:
```python
from openpi_client import websocket_client_policy
client = websocket_client_policy.WebsocketClientPolicy(host="<gpu_host_ip>", port=8000)
action_chunk = client.infer(observation)["actions"]
```

Real-robot inference scripts (with temporal smoothing, RTC, etc.) live in `train_deploy_alignment/inference/`.

---

## Testing Strategy

- **Framework**: pytest
- **Test discovery**: `src`, `scripts`, `packages` (see `pyproject.toml` `[tool.pytest.ini_options]`)
- **Markers**:
  - `manual` — tests that should be run manually (excluded from CI)
- **Run tests**:
  ```bash
  uv run pytest --strict-markers -m "not manual"
  ```
- **Test files**: Co-located with source as `*_test.py` (e.g., `src/openpi/models/pi0_test.py`).
- **GPU fallback**: `src/openpi/conftest.py` auto-sets `JAX_PLATFORMS=cpu` when no GPU is available, so tests can run on CPU-only machines.

### CI

GitHub Actions (`.github/workflows/`):
- `pre-commit.yml` — runs `ruff` and `uv-lock` on PRs/pushes to `main`
- `test.yml` — runs the full pytest suite on a self-hosted `openpi-verylarge` runner

---

## Code Style Guidelines

The project uses **Ruff** for both linting and formatting.

- **Line length**: 120
- **Target Python**: 3.11
- **Import style**: Force single-line imports, sorted within sections (`force-single-line = true`, `force-sort-within-sections = true`)
- **Known third-party**: `wandb` is treated as third-party
- **Excluded directories**: `docker`, `third_party`, `src/openpi/models_pytorch/transformers_replace/*`

Run linting/formatting:
```bash
uv run ruff check .        # lint
uv run ruff format .       # format
pre-commit run --all-files # all hooks
```

### Coding Conventions

- **Type annotations**: Heavy use of `jaxtyping` array shapes and `beartype` runtime checking.
- **Dataclasses**: Configs and policy metadata are frozen dataclasses.
- **Array typing**: Custom module `openpi.shared.array_typing` (aliased as `at`) provides typed JAX array annotations.
- **Logging**: Custom formatter in `scripts/train.py` prints `[I]`, `[W]`, `[E]` prefixes with file:line info.
- **Print statements**: Allowed (`T201` is ignored in Ruff config).

---

## Data Format

Datasets are stored in **LeRobot v2.1** format:
- `data/chunk-XXX/episode_XXXXXX.parquet` — frame-level observations, actions, metadata
- `videos/chunk-XXX/<camera_key>/episode_XXXXXX.mp4` — compressed video per camera
- `meta/info.json`, `meta/episodes.jsonl`, `meta/tasks.jsonl` — dataset metadata

Camera keys used by kai0 configs:
- `observation.images.top_head`
- `observation.images.hand_left`
- `observation.images.hand_right`

Action/state shape: `[N, 14]` (left arm 6-DOF + gripper, right arm 6-DOF + gripper).

---

## Security Considerations

- No secrets should be committed. `.env` and `.envrc` are gitignored.
- `wandb` logging is enabled by default; set `WANDB_MODE=offline` or disable in config if needed.
- Checkpoints and datasets can be large; the default `checkpoint_base_dir` points to an internal shared mount. Override it in your config or via environment variables.
- Docker containers run with `--gpus=all` and host networking; ensure the deployment environment is trusted.

---

## Common Tasks for Agents

### Adding a New Training Config
1. Define a new `TrainConfig(...)` in `src/openpi/training/config.py` and append it to `_CONFIGS`.
2. Ensure `name` is unique.
3. Set `repo_id` to the absolute path of your dataset.
4. Set `weight_loader` to the base checkpoint path.
5. Run `compute_norm_states_fast.py` before training.

### Adding a New Robot Policy
1. Create a new file in `src/openpi/policies/` (follow `agilex_policy.py` or `arx_policy.py`).
2. Implement input/output transform classes.
3. Register the policy in `src/openpi/training/config.py` via a new data-config factory.

### Modifying Model Architecture
- JAX models: edit files in `src/openpi/models/`. `pi0.py` contains the main π₀ architecture; `pi0_config.py` defines hyperparameters.
- PyTorch models: edit files in `src/openpi/models_pytorch/`.
- Be careful with shape annotations (`jaxtyping`) and runtime type checking (`beartype`).

### Working with Checkpoints
- JAX checkpoints are Orbax directories (e.g., `.../90000/` with `params/` inside).
- PyTorch checkpoints are directories containing `model.safetensors`.
- Norm stats (`norm_stats.json`) are saved alongside checkpoints and are required for inference.

---

## Documentation References

- `README.md` — High-level project overview, quick-start, and module descriptions.
- `CLAUDE.md` — Concise cheat-sheet for common commands (similar to this file but shorter).
- `docs/dataset.md` — Dataset structure, download instructions, and LeRobot loading examples.
- `docs/norm_stats_fast.md` — Detailed usage of `compute_norm_states_fast.py`.
- `docs/tda_remote_inference.md` — Remote inference server and client setup.
- `model_arithmetic/README.md` — Full checkpoint-mixing guide.
- `stage_advantage/README.md` — 5-step advantage pipeline (annotate → train → predict → discretize → AWBC).
- `train_deploy_alignment/README.md` — Pointer to sub-module READMEs for data augmentation, DAgger, and inference.
- `setup/README.md` — Hardware setup, camera placement, and 3D-printed part descriptions.
