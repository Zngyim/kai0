# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

χ₀ (kai0) is a resource-efficient framework for robotic manipulation built on top of [openpi](https://github.com/Physical-Intelligence/openpi). It addresses distributional inconsistencies between training and deployment through three modules:

- **Model Arithmetic**: Weight-space merging of trained checkpoints (`model_arithmetic/`)
- **Stage Advantage**: Stage-aware advantage estimation for policy training (`stage_advantage/`)
- **Train-Deploy Alignment**: Data augmentation, DAgger collection, and inference utilities (`train_deploy_alignment/`)

The codebase uses π₀/π₀.5 vision-language-action models for bimanual manipulation (cloth folding, hanging, etc.).

## Key Commands

### Environment Setup
```bash
git clone --recurse-submodules <repo_url>
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### Data & Checkpoints
```bash
python scripts/download_dataset.py                      # Download Task_A/B/C data to ./data
python scripts/download_checkpoints.py                  # Download best models to ./checkpoints
python scripts/download_checkpoints.py --tasks Task_A   # Download specific task checkpoint
```

### Training (JAX/Flax)
```bash
# Compute normalization stats (fast version recommended)
uv run python scripts/compute_norm_states_fast.py --config-name <config_name>

# Full fine-tuning
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp_name=<exp_name>
```

Key configs defined in `src/openpi/training/config.py`: `pi05_flatten_fold_normal`, `pi05_flatten_fold_awbc`, `pi05_hang_cloth_normal`, etc.

### Advantage Estimator Training (PyTorch)
```bash
uv run python scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD --exp_name=run1 --save_interval 10000

# Multi-GPU DDP
uv run torchrun --standalone --nproc_per_node=8 scripts/train_pytorch.py <config_name> --exp_name=run1
```

### Model Arithmetic (Checkpoint Merging)
```bash
# Dump validation data
python model_arithmetic/dump_data.py --dataset <config_name> --output val.pkl

# Mix checkpoints (JAX)
python model_arithmetic/arithmetic.py --config <config_name> --data-path val.pkl \
  --checkpoints /path/to/ckpt1 /path/to/ckpt2 /path/to/ckpt3 \
  --output /path/to/mixed --optimize_method inverse_loss --use_gpu --gpu_ids "0"
```

Methods: `average`, `inverse_loss`, `gradient_descent`, `adaptive_gradient_descent`, `greedy`, or manual `--weights`.

### Inference Server
```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config_name> --policy.dir=<checkpoint_dir> [--port=8000]
```

### Testing & Linting
```bash
uv run pytest --strict-markers -m "not manual"   # Run tests (markers defined in pyproject.toml)
uv run ruff check .                              # Lint
uv run ruff format .                             # Format
pre-commit run --all-files                       # Run all hooks
```

## Architecture

### Training Pipeline
- `scripts/train.py`: Main JAX training loop (uses `src/openpi/training/`)
- `scripts/train_pytorch.py`: PyTorch training for advantage estimator
- `src/openpi/models/`: JAX model definitions (π₀, π₀.5, Gemma, SigLIP, ViT)
- `src/openpi/models_pytorch/`: PyTorch model implementations
- `src/openpi/policies/`: Robot-specific policy configs (agilex, arx, aloha, etc.)
- `src/openpi/training/config.py`: Central config registry (`_CONFIGS` dict)

### Data Format
- Uses [LeRobot](https://github.com/huggingface/lerobot) dataset format (parquet + video chunks + meta)
- Dataset path set via `repo_id` in config (e.g., `<repo_root>/data/Task_A/base`)
- Normalization stats in `norm_stats.json` alongside checkpoint

### Client Package
- `packages/openpi-client/`: WebSocket client for inference
- Runtime agent in `runtime/agent.py`, environment in `runtime/environment.py`

### Config Management
Edit `src/openpi/training/config.py` (lines ~1173–1226) to set:
- `repo_id`: Absolute path to dataset
- `weight_loader`: Path to π₀.5 base checkpoint

### Module-Specific Docs
- `model_arithmetic/README.md`: Dataset splitting, checkpoint mixing methods
- `stage_advantage/README.md`: 5-step pipeline (annotate → train estimator → predict → discretize → AWBC)
- `train_deploy_alignment/README.md`: Data augmentation, DAgger, inference (temporal smoothing, RTC)

## Inference on Robots

Two-machine setup: GPU host runs `serve_policy.py`, robot IPC runs platform-specific client.

- **Agilex Piper**: `train_deploy_alignment/inference/agilex/` — temporal smoothing mode
- **ARX X5**: `train_deploy_alignment/inference/arx/` — RTC (real-time chunking) mode

Set `--host <gpu_host_ip>` in the client script to connect to the policy server.

## AWBC Training

For Advantage-Weighted Behavior Cloning, the advantage dataset (`Task_A/advantage/`) has:
- `task_index` column in parquet (discretized advantage label)
- `meta/tasks.jsonl` mapping task_index → prompt string

At inference, use the same prompt format (e.g., `"fold the cloth, Advantage: positive"`) as training.

## Dependencies

- Python ≥3.11
- JAX 0.5.3 with CUDA 12 support
- PyTorch 2.7.1 (for advantage estimator)
- Flax 0.10.2, Orbax checkpoint 0.11.13
- lerobot (from HuggingFace git repo)
- Uses [uv](https://docs.astral.sh/uv/) for package management

我只关心有关于 advantage 模块的内容。