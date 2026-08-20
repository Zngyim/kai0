#!/usr/bin/env python3
"""Visualize a two-view KAI0 advantage checkpoint with training-identical inputs.

The model input is built with the exact transform chain used by
``ADVANTAGE_TORCH_KAI0_TOWEL_FOLD``. In particular, the first frame is placed
in the ``-100`` reference slots and the current frame is placed in the ``0``
slots; the ``-100`` name denotes the comparison observation, not a fixed
temporal offset.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

# JAX is used by the training transforms. Keep it on CPU so it does not reserve
# memory on the GPU shared with PyTorch inference and the ongoing DDP run.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import cv2  # noqa: E402
import jax  # noqa: E402
from lerobot.common.datasets.video_utils import decode_video_frames  # noqa: E402
import matplotlib as mpl  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import safetensors.torch  # noqa: E402
import torch  # noqa: E402

from openpi.models import model as model_lib  # noqa: E402
from openpi.models_pytorch import preprocessing_pytorch  # noqa: E402
from openpi.models_pytorch.pi0_pytorch import AdvantageEstimator  # noqa: E402
from openpi.training import config as config_lib  # noqa: E402
from openpi.training import data_loader  # noqa: E402
import openpi.transforms as transforms  # noqa: E402

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOGGER = logging.getLogger("advantage_visualization")

DEFAULT_CONFIG_NAME = "ADVANTAGE_TORCH_KAI0_TOWEL_FOLD"
DEFAULT_CHECKPOINT = Path(
    "/mnt/pfs/zhangjiyao/yiming/checkpoints/ADVANTAGE_TORCH_KAI0_TOWEL_FOLD/advantage_towel_2gpu"
)
DEFAULT_DATASET = project_root / "data/stage_advantage/zngyim_dataset_linear"
DEFAULT_OUTPUT_DIR = project_root / "eval_viz_towel_fold_advantage_5000"

FRONT_VIDEO_KEY = "extra_view_image"
WRIST_VIDEO_KEY = "image"
EXPECTED_OBSERVATION_KEYS = (
    "base_-100_rgb",
    "left_wrist_-100_rgb",
    "base_0_rgb",
    "left_wrist_0_rgb",
)


@dataclasses.dataclass(frozen=True)
class EpisodePaths:
    episode_index: int
    parquet: Path
    front_video: Path
    wrist_video: Path


@dataclasses.dataclass(frozen=True)
class EpisodeData:
    timestamps: np.ndarray
    frame_indices: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    stage_progress_gt: np.ndarray

    @property
    def num_frames(self) -> int:
        return len(self.timestamps)


def resolve_checkpoint(checkpoint_path: Path) -> Path:
    """Resolve either a checkpoint step directory or an experiment directory."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if (checkpoint_path / "model.safetensors").is_file():
        return checkpoint_path

    candidates = [
        child
        for child in checkpoint_path.iterdir()
        if child.is_dir() and child.name.isdigit() and (child / "model.safetensors").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No model.safetensors checkpoint found under {checkpoint_path}")
    return max(candidates, key=lambda path: int(path.name))


def read_dataset_info(dataset_dir: Path) -> dict[str, Any]:
    info_path = dataset_dir / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {info_path}")
    with info_path.open() as file:
        return json.load(file)


def get_episode_paths(dataset_dir: Path, episode_index: int, info: dict[str, Any]) -> EpisodePaths:
    chunks_size = int(info["chunks_size"])
    chunk_index = episode_index // chunks_size
    data_template = info["data_path"]
    video_template = info["video_path"]
    parquet = dataset_dir / data_template.format(
        episode_chunk=chunk_index,
        episode_index=episode_index,
    )
    front_video = dataset_dir / video_template.format(
        episode_chunk=chunk_index,
        episode_index=episode_index,
        video_key=FRONT_VIDEO_KEY,
    )
    wrist_video = dataset_dir / video_template.format(
        episode_chunk=chunk_index,
        episode_index=episode_index,
        video_key=WRIST_VIDEO_KEY,
    )
    return EpisodePaths(episode_index, parquet, front_video, wrist_video)


def select_episode(
    dataset_dir: Path,
    *,
    seed: int,
    episode_index: int | None,
) -> tuple[EpisodePaths, dict[str, Any]]:
    """Select a valid episode, deterministically when no explicit index is given."""
    info = read_dataset_info(dataset_dir)
    total_episodes = int(info["total_episodes"])
    if episode_index is not None:
        if not 0 <= episode_index < total_episodes:
            raise ValueError(f"episode_index must be in [0, {total_episodes}), got {episode_index}")
        paths = get_episode_paths(dataset_dir, episode_index, info)
        if not (paths.parquet.is_file() and paths.front_video.is_file() and paths.wrist_video.is_file()):
            raise FileNotFoundError(f"Missing parquet or UMI video for episode {episode_index} in {dataset_dir}")
        return paths, info

    valid_paths = []
    for index in range(total_episodes):
        paths = get_episode_paths(dataset_dir, index, info)
        if paths.parquet.is_file() and paths.front_video.is_file() and paths.wrist_video.is_file():
            valid_paths.append(paths)
    if not valid_paths:
        raise FileNotFoundError(f"Could not find a complete episode among {total_episodes} episodes in {dataset_dir}")
    return random.Random(seed).choice(valid_paths), info


def load_episode_data(parquet_path: Path) -> EpisodeData:
    table = pq.read_table(
        parquet_path,
        columns=["timestamp", "frame_index", "state", "actions", "stage_progress_gt"],
    )
    return EpisodeData(
        timestamps=np.asarray(table["timestamp"].to_pylist(), dtype=np.float64),
        frame_indices=np.asarray(table["frame_index"].to_pylist(), dtype=np.int64),
        states=np.asarray(table["state"].to_pylist(), dtype=np.float32),
        actions=np.asarray(table["actions"].to_pylist(), dtype=np.float32),
        stage_progress_gt=np.asarray(table["stage_progress_gt"].to_pylist(), dtype=np.float32),
    )


def compute_relative_ground_truth(stage_progress_gt: np.ndarray) -> np.ndarray:
    if stage_progress_gt.ndim != 1 or len(stage_progress_gt) == 0:
        raise ValueError("stage_progress_gt must be a non-empty 1D array")
    return stage_progress_gt.astype(np.float32, copy=False) - np.float32(stage_progress_gt[0])


def make_action_chunk(actions: np.ndarray, frame_index: int, action_horizon: int) -> np.ndarray:
    """Match LeRobot delta-index clamping at the end of an episode."""
    indices = np.minimum(np.arange(frame_index, frame_index + action_horizon), len(actions) - 1)
    return actions[indices]


def build_training_transform(train_config: config_lib.TrainConfig):
    """Build the same ordered transform chain used by create_torch_data_loader."""
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if train_config.skip_norm_stats:
        norm_stats = {}
    else:
        norm_stats = data_config.norm_stats
        if norm_stats is None:
            raise ValueError("Normalization stats are required by this training config")

    transform = transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    return data_config, transform


def build_raw_pair(
    *,
    current_front: torch.Tensor,
    current_wrist: torch.Tensor,
    reference_front: torch.Tensor,
    reference_wrist: torch.Tensor,
    state: np.ndarray,
    action_chunk: np.ndarray,
    episode_length: int,
    frame_index: int,
    episode_index: int,
    progress: float,
) -> dict[str, Any]:
    """Build the pre-repack item expected by the towel advantage training config."""
    return {
        "extra_view_image": current_front,
        "image": current_wrist,
        "his_-100_extra_view_image": reference_front,
        "his_-100_image": reference_wrist,
        "state": state,
        "actions": action_chunk,
        "episode_length": episode_length,
        "frame_index": frame_index,
        "episode_index": episode_index,
        "progress": np.float32(progress),
    }


def collate_like_training(items: list[dict[str, Any]]) -> model_lib.Observation:
    """Reproduce the training TorchDataLoader collation and Observation conversion."""
    batch = jax.tree.map(
        lambda *values: np.stack([np.asarray(value) for value in values], axis=0),
        *items,
    )
    batch = jax.tree.map(torch.as_tensor, batch)
    return model_lib.Observation.from_dict(batch)


def move_observation(observation: model_lib.Observation, device: torch.device) -> model_lib.Observation:
    return jax.tree.map(lambda value: value.to(device), observation)


def assert_model_input_contract(observation: model_lib.Observation) -> None:
    """Assert the final ordering and values seen by AdvantageEstimator.embed_prefix."""
    processed = preprocessing_pytorch.preprocess_observation_pytorch_custom(
        observation,
        train=False,
        return_full_obs=False,
        apply_aug=False,
    )
    actual_keys = tuple(processed.images)
    if actual_keys != EXPECTED_OBSERVATION_KEYS:
        raise AssertionError(f"Unexpected model image order: {actual_keys}; expected {EXPECTED_OBSERVATION_KEYS}")
    if any(tuple(image.shape[1:]) != (3, 224, 224) for image in processed.images.values()):
        raise AssertionError("All model images must have shape [B, 3, 224, 224]")
    if any(not bool(mask.all()) for mask in processed.image_masks.values()):
        raise AssertionError("All four UMI image masks must be true")
    if tuple(processed.state.shape[1:]) != (32,) or not bool(torch.all(processed.state == 0)):
        raise AssertionError("UMI state must be masked to zero and padded to 32 dimensions")
    if tuple(processed.tokenized_prompt.shape[1:]) != (200,):
        raise AssertionError("Tokenized prompt must have length 200")


def _compare_arrays(name: str, expected: Any, actual: Any) -> None:
    expected_array = np.asarray(expected)
    actual_array = np.asarray(actual)
    if expected_array.shape != actual_array.shape or not np.array_equal(expected_array, actual_array):
        max_diff = float(np.max(np.abs(expected_array.astype(np.float64) - actual_array.astype(np.float64))))
        raise AssertionError(f"Training parity mismatch for {name}: max_abs_diff={max_diff}")


def verify_training_parity(
    train_config: config_lib.TrainConfig,
    dataset_dir: Path,
    data_transform,
) -> None:
    """Compare a production-built frame pair against the real training dataset path."""
    LOGGER.info("Running golden training/inference input parity check")
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    raw_dataset = data_loader.create_advantage_torch_dataset(
        data_config,
        train_config.model.action_horizon,
        train_config.model,
        train_config,
    )

    random_state = random.getstate()
    try:
        random.seed(123)
        training_raw = raw_dataset[0]
    finally:
        random.setstate(random_state)

    current_frame = int(training_raw["frame_index"])
    reference_frame = int(training_raw["his_-100_frame_index"])
    manual_raw = build_raw_pair(
        current_front=training_raw[FRONT_VIDEO_KEY],
        current_wrist=training_raw[WRIST_VIDEO_KEY],
        reference_front=training_raw[f"his_-100_{FRONT_VIDEO_KEY}"],
        reference_wrist=training_raw[f"his_-100_{WRIST_VIDEO_KEY}"],
        state=np.asarray(training_raw["state"], dtype=np.float32),
        action_chunk=np.asarray(training_raw["actions"], dtype=np.float32),
        episode_length=int(training_raw["episode_length"]),
        frame_index=current_frame,
        episode_index=int(training_raw["episode_index"]),
        progress=float(training_raw["progress"]),
    )

    training_transformed = data_transform(training_raw)
    manual_transformed = data_transform(manual_raw)
    for key in training_transformed["image"]:
        _compare_arrays(f"image/{key}", training_transformed["image"][key], manual_transformed["image"][key])
        _compare_arrays(
            f"image_mask/{key}",
            training_transformed["image_mask"][key],
            manual_transformed["image_mask"][key],
        )
    for key in ("state", "tokenized_prompt", "tokenized_prompt_mask", "progress"):
        _compare_arrays(key, training_transformed[key], manual_transformed[key])

    # Confirm that the production decoder returns the same tensors as the
    # AdvantageLerobotDataset pyav path for both current and reference frames.
    paths = get_episode_paths(dataset_dir, int(training_raw["episode_index"]), read_dataset_info(dataset_dir))
    parquet = pq.read_table(paths.parquet, columns=["timestamp"])
    current_timestamp = float(parquet["timestamp"][current_frame].as_py())
    reference_timestamp = float(parquet["timestamp"][reference_frame].as_py())
    for path, current_key, reference_key in (
        (paths.front_video, FRONT_VIDEO_KEY, f"his_-100_{FRONT_VIDEO_KEY}"),
        (paths.wrist_video, WRIST_VIDEO_KEY, f"his_-100_{WRIST_VIDEO_KEY}"),
    ):
        decoded = decode_video_frames(
            path,
            [current_timestamp, reference_timestamp],
            tolerance_s=raw_dataset.tolerance_s,
            backend="pyav",
        )
        _compare_arrays(f"decoded/{current_key}", training_raw[current_key], decoded[0])
        _compare_arrays(f"decoded/{reference_key}", training_raw[reference_key], decoded[1])

    parity_observation = collate_like_training([manual_transformed])
    assert_model_input_contract(parity_observation)
    LOGGER.info(
        "Golden parity passed: current_frame=%d reference_frame=%d keys=%s",
        current_frame,
        reference_frame,
        EXPECTED_OBSERVATION_KEYS,
    )


def load_model(
    train_config: config_lib.TrainConfig,
    checkpoint_dir: Path,
    device: torch.device,
) -> AdvantageEstimator:
    model = AdvantageEstimator(train_config.model).to(device)
    model.eval()
    model_path = checkpoint_dir / "model.safetensors"
    LOGGER.info("Strictly loading checkpoint: %s", model_path)
    safetensors.torch.load_model(model, model_path, strict=True)
    return model


def infer_episode(
    *,
    model: AdvantageEstimator,
    device: torch.device,
    train_config: config_lib.TrainConfig,
    data_transform,
    paths: EpisodePaths,
    episode_data: EpisodeData,
    ground_truth: np.ndarray,
    batch_size: int,
    decode_chunk_size: int,
) -> tuple[np.ndarray, int]:
    if batch_size <= 0 or decode_chunk_size <= 0:
        raise ValueError("batch_size and decode_chunk_size must be positive")

    predictions = np.full(episode_data.num_frames, np.nan, dtype=np.float32)
    predictions[0] = 0.0
    reference_timestamp = float(episode_data.timestamps[0])
    reference_front = decode_video_frames(
        paths.front_video,
        [reference_timestamp],
        tolerance_s=1e-4,
        backend="pyav",
    )[0]
    reference_wrist = decode_video_frames(
        paths.wrist_video,
        [reference_timestamp],
        tolerance_s=1e-4,
        backend="pyav",
    )[0]

    effective_batch_size = batch_size
    contract_checked = False
    for chunk_start in range(0, episode_data.num_frames, decode_chunk_size):
        chunk_end = min(chunk_start + decode_chunk_size, episode_data.num_frames)
        chunk_timestamps = episode_data.timestamps[chunk_start:chunk_end].tolist()
        front_frames = decode_video_frames(
            paths.front_video,
            chunk_timestamps,
            tolerance_s=1e-4,
            backend="pyav",
        )
        wrist_frames = decode_video_frames(
            paths.wrist_video,
            chunk_timestamps,
            tolerance_s=1e-4,
            backend="pyav",
        )

        local_start = 0
        if chunk_start == 0:
            local_start = 1
        while local_start < chunk_end - chunk_start:
            current_batch_size = min(effective_batch_size, chunk_end - chunk_start - local_start)
            global_start = chunk_start + local_start
            items = []
            for offset in range(current_batch_size):
                frame_index = global_start + offset
                items.append(
                    data_transform(
                        build_raw_pair(
                            current_front=front_frames[local_start + offset],
                            current_wrist=wrist_frames[local_start + offset],
                            reference_front=reference_front,
                            reference_wrist=reference_wrist,
                            state=episode_data.states[frame_index],
                            action_chunk=make_action_chunk(
                                episode_data.actions,
                                frame_index,
                                train_config.model.action_horizon,
                            ),
                            episode_length=episode_data.num_frames,
                            frame_index=int(episode_data.frame_indices[frame_index]),
                            episode_index=paths.episode_index,
                            progress=float(ground_truth[frame_index]),
                        )
                    )
                )

            observation = collate_like_training(items)
            if not contract_checked:
                assert_model_input_contract(observation)
                contract_checked = True
            observation = move_observation(observation, device)

            try:
                with torch.inference_mode():
                    values = model.sample_values(device, observation)[:, 0]
                predictions[global_start : global_start + current_batch_size] = (
                    values.float().cpu().numpy().astype(np.float32)
                )
                local_start += current_batch_size
            except torch.cuda.OutOfMemoryError:
                del observation
                torch.cuda.empty_cache()
                if effective_batch_size == 1:
                    raise
                effective_batch_size = max(1, effective_batch_size // 2)
                LOGGER.warning("CUDA OOM; retrying with batch_size=%d", effective_batch_size)

        LOGGER.info(
            "Inference progress: %d/%d frames (batch_size=%d)",
            chunk_end,
            episode_data.num_frames,
            effective_batch_size,
        )

    if not np.isfinite(predictions).all():
        missing = np.flatnonzero(~np.isfinite(predictions))
        raise RuntimeError(f"Missing predictions for frames: {missing[:10].tolist()}")
    return predictions, effective_batch_size


def compute_metrics(ground_truth: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    error = predictions - ground_truth
    correlation = float(np.corrcoef(ground_truth, predictions)[0, 1])
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "pearson_correlation": correlation,
        "max_absolute_error": float(np.max(np.abs(error))),
    }


def save_curve(
    output_path: Path,
    timestamps: np.ndarray,
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    metrics: dict[str, float],
    episode_index: int,
) -> None:
    elapsed = timestamps - timestamps[0]
    error = predictions - ground_truth
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})

    axes[0].plot(elapsed, ground_truth, color="#1565C0", linewidth=2.2, label="GT: progress(t) - progress(0)")
    axes[0].plot(elapsed, predictions, color="#EF6C00", linewidth=1.8, label="Prediction: model(frame 0, frame t)")
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_ylabel("Advantage relative to first frame")
    axes[0].grid(visible=True, alpha=0.25)
    axes[0].legend(loc="upper left")
    axes[0].set_title(
        f"Towel-fold training trajectory — episode {episode_index:06d}\n"
        f"MAE={metrics['mae']:.4f}   RMSE={metrics['rmse']:.4f}   "
        f"Pearson r={metrics['pearson_correlation']:.4f}"
    )

    axes[1].plot(elapsed, error, color="#C62828", linewidth=1.5, label="Prediction - GT")
    axes[1].fill_between(elapsed, 0.0, error, color="#EF9A9A", alpha=0.35)
    axes[1].axhline(0.0, color="black", linewidth=0.9)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Error")
    axes[1].grid(visible=True, alpha=0.25)
    axes[1].legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def overlay_values(
    front_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    *,
    frame_index: int,
    timestamp: float,
    ground_truth: float,
    prediction: float,
) -> np.ndarray:
    """Create a clear side-by-side BGR frame with top-right values."""
    if front_rgb.shape != wrist_rgb.shape:
        wrist_rgb = cv2.resize(wrist_rgb, (front_rgb.shape[1], front_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    canvas_rgb = np.concatenate([front_rgb, wrist_rgb], axis=1)
    frame = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Front view", (14, 32), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        "Left wrist view",
        (front_rgb.shape[1] + 14, 32),
        font,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    error = prediction - ground_truth
    lines = (
        (f"Frame {frame_index:04d}   Time {timestamp:6.2f}s", (255, 255, 255)),
        (f"GT Advantage    {ground_truth:+.4f}", (255, 220, 80)),
        (f"Pred Advantage  {prediction:+.4f}", (40, 180, 255)),
        (f"Error           {error:+.4f}", (120, 255, 120)),
    )
    font_scale = 0.68
    thickness = 2
    margin = 14
    line_height = 31
    box_width = 430
    box_height = margin * 2 + line_height * len(lines)
    x0 = frame.shape[1] - box_width - margin
    y0 = margin
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (frame.shape[1] - margin, y0 + box_height), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0.0, frame)
    cv2.rectangle(frame, (x0, y0), (frame.shape[1] - margin, y0 + box_height), (210, 210, 210), 1)
    for line_index, (text, color) in enumerate(lines):
        y = y0 + margin + 22 + line_index * line_height
        cv2.putText(frame, text, (x0 + margin, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


class FfmpegWriter:
    def __init__(self, output_path: Path, *, width: int, height: int, fps: float):
        self.output_path = output_path
        self.temp_path = output_path.with_name(f".{output_path.stem}.tmp.mp4")
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.8f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(self.temp_path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        stderr = self.process.stderr.read().decode() if self.process.stderr is not None else ""
        return_code = self.process.wait()
        if return_code != 0:
            with contextlib.suppress(FileNotFoundError):
                self.temp_path.unlink()
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr}")
        self.temp_path.replace(self.output_path)

    def __enter__(self) -> FfmpegWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            if self.process.stdin is not None:
                self.process.stdin.close()
            self.process.terminate()
            self.process.wait()
            with contextlib.suppress(FileNotFoundError):
                self.temp_path.unlink()


def save_overlay_video(
    output_path: Path,
    *,
    paths: EpisodePaths,
    timestamps: np.ndarray,
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    fps: float,
    decode_chunk_size: int,
) -> None:
    first_front = decode_video_frames(
        paths.front_video,
        [float(timestamps[0])],
        tolerance_s=1e-4,
        backend="pyav",
    )[0]
    height, width = int(first_front.shape[1]), int(first_front.shape[2])

    with FfmpegWriter(output_path, width=width * 2, height=height, fps=fps) as writer:
        for chunk_start in range(0, len(timestamps), decode_chunk_size):
            chunk_end = min(chunk_start + decode_chunk_size, len(timestamps))
            chunk_timestamps = timestamps[chunk_start:chunk_end].tolist()
            front_frames = decode_video_frames(
                paths.front_video,
                chunk_timestamps,
                tolerance_s=1e-4,
                backend="pyav",
            )
            wrist_frames = decode_video_frames(
                paths.wrist_video,
                chunk_timestamps,
                tolerance_s=1e-4,
                backend="pyav",
            )
            for local_index in range(chunk_end - chunk_start):
                frame_index = chunk_start + local_index
                front_rgb = (
                    front_frames[local_index].mul(255).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
                )
                wrist_rgb = (
                    wrist_frames[local_index].mul(255).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
                )
                writer.write(
                    overlay_values(
                        front_rgb,
                        wrist_rgb,
                        frame_index=int(frame_index),
                        timestamp=float(timestamps[frame_index] - timestamps[0]),
                        ground_truth=float(ground_truth[frame_index]),
                        prediction=float(predictions[frame_index]),
                    )
                )
            LOGGER.info("Video rendering progress: %d/%d frames", chunk_end, len(timestamps))


def save_values_csv(
    output_path: Path,
    episode_data: EpisodeData,
    ground_truth: np.ndarray,
    predictions: np.ndarray,
) -> None:
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame_index", "timestamp", "elapsed_seconds", "gt_advantage", "pred_advantage", "error"])
        for index in range(episode_data.num_frames):
            writer.writerow(
                [
                    int(episode_data.frame_indices[index]),
                    f"{episode_data.timestamps[index]:.9f}",
                    f"{episode_data.timestamps[index] - episode_data.timestamps[0]:.9f}",
                    f"{ground_truth[index]:.9f}",
                    f"{predictions[index]:.9f}",
                    f"{predictions[index] - ground_truth[index]:.9f}",
                ]
            )


def validate_episode(paths: EpisodePaths, episode_data: EpisodeData, info: dict[str, Any]) -> float:
    if episode_data.num_frames < 2:
        raise ValueError("Selected episode must contain at least two frames")
    if not np.array_equal(episode_data.frame_indices, np.arange(episode_data.num_frames)):
        raise ValueError("frame_index must be contiguous and start at zero")
    if not np.all(np.diff(episode_data.timestamps) > 0):
        raise ValueError("timestamps must be strictly increasing")

    fps = float(info["fps"])
    for video_path in (paths.front_video, paths.wrist_video):
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")
            video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        if video_frames != episode_data.num_frames:
            raise ValueError(f"Frame count mismatch for {video_path}: video={video_frames}, parquet={episode_data.num_frames}")
        if abs(video_fps - fps) > 1e-4:
            raise ValueError(f"FPS mismatch for {video_path}: video={video_fps}, metadata={fps}")
    return fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decode-chunk-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--verify-training-parity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare one input against the real training DataLoader before inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

    checkpoint_dir = resolve_checkpoint(args.ckpt_dir)
    dataset_dir = args.dataset.expanduser().resolve()
    paths, info = select_episode(
        dataset_dir,
        seed=args.seed,
        episode_index=args.episode_index,
    )
    episode_data = load_episode_data(paths.parquet)
    fps = validate_episode(paths, episode_data, info)
    ground_truth = compute_relative_ground_truth(episode_data.stage_progress_gt)
    train_config = config_lib.get_config(args.config_name)
    _, data_transform = build_training_transform(train_config)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{paths.episode_index:06d}"
    curve_path = output_dir / f"{stem}_curve.png"
    video_path = output_dir / f"{stem}_overlay.mp4"
    csv_path = output_dir / f"{stem}_values.csv"
    metadata_path = output_dir / f"{stem}_metadata.json"

    LOGGER.info(
        "Selected episode %06d: %d frames, %.2f seconds, seed=%d",
        paths.episode_index,
        episode_data.num_frames,
        episode_data.num_frames / fps,
        args.seed,
    )
    if args.verify_training_parity:
        verify_training_parity(train_config, dataset_dir, data_transform)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this script in the host GPU environment.")
    model = load_model(train_config, checkpoint_dir, device)
    predictions, effective_batch_size = infer_episode(
        model=model,
        device=device,
        train_config=train_config,
        data_transform=data_transform,
        paths=paths,
        episode_data=episode_data,
        ground_truth=ground_truth,
        batch_size=args.batch_size,
        decode_chunk_size=args.decode_chunk_size,
    )
    metrics = compute_metrics(ground_truth, predictions)

    save_curve(
        curve_path,
        episode_data.timestamps,
        ground_truth,
        predictions,
        metrics,
        paths.episode_index,
    )
    save_values_csv(csv_path, episode_data, ground_truth, predictions)
    save_overlay_video(
        video_path,
        paths=paths,
        timestamps=episode_data.timestamps,
        ground_truth=ground_truth,
        predictions=predictions,
        fps=fps,
        decode_chunk_size=args.decode_chunk_size,
    )

    checkpoint_file = checkpoint_dir / "model.safetensors"
    metadata = {
        "config_name": args.config_name,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_step": int(checkpoint_dir.name) if checkpoint_dir.name.isdigit() else None,
        "checkpoint_size_bytes": checkpoint_file.stat().st_size,
        "dataset": str(dataset_dir),
        "episode_index": paths.episode_index,
        "selection_seed": args.seed,
        "num_frames": episode_data.num_frames,
        "fps": fps,
        "duration_seconds": episode_data.num_frames / fps,
        "prompt": train_config.data.default_prompt,
        "ground_truth_definition": "stage_progress_gt[t] - stage_progress_gt[0]",
        "prediction_definition": "model(frame_0, frame_t)",
        "camera_mapping": {
            "base_-100_rgb": f"{FRONT_VIDEO_KEY}[0]",
            "left_wrist_-100_rgb": f"{WRIST_VIDEO_KEY}[0]",
            "base_0_rgb": f"{FRONT_VIDEO_KEY}[t]",
            "left_wrist_0_rgb": f"{WRIST_VIDEO_KEY}[t]",
        },
        "model_image_order": list(EXPECTED_OBSERVATION_KEYS),
        "video_backend": "pyav",
        "requested_batch_size": args.batch_size,
        "effective_batch_size": effective_batch_size,
        "decode_chunk_size": args.decode_chunk_size,
        "device": str(device),
        "training_parity_verified": bool(args.verify_training_parity),
        "metrics": metrics,
        "outputs": {
            "curve": str(curve_path),
            "video": str(video_path),
            "values": str(csv_path),
        },
    }
    with metadata_path.open("w") as file:
        json.dump(metadata, file, indent=2)

    LOGGER.info("Metrics: %s", json.dumps(metrics, sort_keys=True))
    LOGGER.info("Curve: %s", curve_path)
    LOGGER.info("Video: %s", video_path)
    LOGGER.info("Values: %s", csv_path)
    LOGGER.info("Metadata: %s", metadata_path)


if __name__ == "__main__":
    main()
