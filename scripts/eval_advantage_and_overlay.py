#!/usr/bin/env python3
"""Evaluate advantage on a LeRobot dataset and overlay predictions on video frames.

Usage:
    cd /mnt/pfs/zhangjiyao/yiming/kai0
    uv run python scripts/eval_advantage_and_overlay.py \
        --ckpt-dir /mnt/pfs/zhangjiyao/yiming/checkpoints/STACK_BLOCKS_ADVANTAGE/run2/10000 \
        --config-name STACK_BLOCKS_ADVANTAGE \
        --dataset /mnt/pfs/zhangjiyao/yiming/kai0/testdata/0710_disrupt \
        --prompt "Stack three blocks." \
        --output-dir ./testdata/0710_disrupt_overlay_run2
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from stage_advantage.annotation.evaluator import SimpleValueEvaluator  # noqa: E402


def build_overlay_values(
    results: list[dict],
    n_frames: int,
    relative_interval: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-frame progress and backward-looking direct relative-advantage arrays."""
    progress_values = np.full(n_frames, np.nan, dtype=np.float32)
    relative_advantages = np.full(n_frames, np.nan, dtype=np.float32)

    for result in results:
        frame_idx = result["frame_idx"]
        if 0 <= frame_idx < n_frames:
            progress_values[frame_idx] = result.get("absolute_value", np.nan)

        future_frame_idx = result["future_frame_idx"]
        if future_frame_idx - frame_idx == relative_interval and 0 <= future_frame_idx < n_frames:
            relative_advantages[future_frame_idx] = result.get("relative_advantage", np.nan)

    return progress_values, relative_advantages


def compute_stall_mask(
    relative_advantages: np.ndarray,
    window: int = 10,
    threshold: float = 0.08,
) -> np.ndarray:
    """Return True when a finite relative-advantage window is entirely below the threshold."""
    n = len(relative_advantages)
    mask = np.zeros(n, dtype=bool)
    for i in range(window - 1, n):
        window_values = relative_advantages[i - window + 1 : i + 1]
        if np.all(np.isfinite(window_values)) and np.all(window_values < threshold):
            mask[i] = True
    return mask


def apply_red_overlay(frame: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Apply a semi-transparent red overlay to the whole frame."""
    red = np.full_like(frame, (0, 0, 255))
    return cv2.addWeighted(frame, 1 - alpha, red, alpha, 0)


def overlay_text_on_frame(
    frame: np.ndarray,
    progress_value: float,
    relative_advantage: float,
    *,
    font_scale: float = 0.6,
    thickness: int = 2,
    text_color: tuple = (0, 255, 0),
    bg_color: tuple = (0, 0, 0),
    show_ood: bool = False,
    ood_color: tuple = (0, 0, 255),
) -> np.ndarray:
    """Draw progress, direct relative advantage, and an optional OOD label."""
    w = frame.shape[1]
    lines = [
        (f"progress: {progress_value:+.3f}", text_color),
        (f"relative_adv: {relative_advantage:+.3f}", text_color),
    ]
    if show_ood:
        lines.append(("OOD", ood_color))

    # Compute text sizes to align to the right
    text_sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0] for line, _ in lines]
    line_height = max(th for _, th in text_sizes) + 10

    margin = 10
    x_right = w - margin
    y_top = margin + line_height

    for i, ((line, color), (tw, th)) in enumerate(zip(lines, text_sizes, strict=True)):
        x = x_right - tw
        y = y_top + i * line_height

        # Draw background rectangle
        pt1 = (x - 4, y - th - 4)
        pt2 = (x_right + 4, y + 4)
        cv2.rectangle(frame, pt1, pt2, bg_color, -1)

        # Draw text
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    return frame


def process_video_with_overlay(
    input_video_path: Path,
    output_video_path: Path,
    progress_values: np.ndarray,
    relative_advantages: np.ndarray,
    fps: float = 30.0,
    playback_speed: float = 0.5,
    font_scale: float = 0.6,
    thickness: int = 2,
    relative_threshold: float = 0.08,
    overlay_alpha: float = 0.3,
):
    """Read a video, overlay progress and relative advantage, and write the result."""
    if playback_speed <= 0:
        raise ValueError(f"playback_speed must be positive, got {playback_speed}")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    output_fps = input_fps * playback_speed
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, output_fps, (frame_width, frame_height))

    stall_mask = compute_stall_mask(relative_advantages, window=10, threshold=relative_threshold)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(progress_values):
            progress_value = progress_values[frame_idx]
            relative_advantage = relative_advantages[frame_idx]
            is_stalled = stall_mask[frame_idx]
            frame = overlay_text_on_frame(
                frame,
                progress_value,
                relative_advantage,
                font_scale=font_scale,
                thickness=thickness,
                show_ood=is_stalled,
            )

            # if is_stalled:
            #     frame = apply_red_overlay(frame, alpha=overlay_alpha)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(
        f"  Written overlay video: {output_video_path} "
        f"({frame_idx} frames, {playback_speed:g}x speed, {output_fps:g} FPS)"
    )


def run_eval_and_overlay(
    ckpt_dir: Path,
    dataset_dir: Path,
    config_name: str,
    prompt: str,
    output_dir: Path,
    batch_size: int = 8,
    playback_speed: float = 0.5,
    font_scale: float = 0.6,
    thickness: int = 2,
    relative_threshold: float = 0.08,
    overlay_alpha: float = 0.3,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load evaluator
    print(f"[INFO] Loading checkpoint from: {ckpt_dir}")
    evaluator = SimpleValueEvaluator(
        config_name=config_name,
        ckpt_dir=str(ckpt_dir),
        num_workers=4,
    )

    # 2. Load dataset metadata from local info.json
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset info not found: {info_path}")
    with open(info_path) as f:
        info = json.load(f)
    total_episodes = info["total_episodes"]
    chunks_size = info["chunks_size"]
    video_path_template = info["video_path"]
    print(f"[INFO] Dataset: {dataset_dir}, Total episodes: {total_episodes}")

    camera_views = [
        ("observation.images.top_head", "top_head"),
        ("observation.images.hand_left", "hand_left"),
        ("observation.images.hand_right", "hand_right"),
    ]

    for ep_idx in range(total_episodes):
        print(f"\n[INFO] Processing episode {ep_idx} ...")

        chunk_idx = ep_idx // chunks_size

        # Build video paths using the template from info.json
        top_video = dataset_dir / video_path_template.format(
            episode_chunk=chunk_idx, episode_index=ep_idx, video_key="observation.images.top_head"
        )
        left_video = dataset_dir / video_path_template.format(
            episode_chunk=chunk_idx, episode_index=ep_idx, video_key="observation.images.hand_left"
        )
        right_video = dataset_dir / video_path_template.format(
            episode_chunk=chunk_idx, episode_index=ep_idx, video_key="observation.images.hand_right"
        )

        if not top_video.exists() or not left_video.exists() or not right_video.exists():
            print(f"  [WARN] Missing video for episode {ep_idx}, skipping")
            continue

        # Read frame count from top video (all should match)
        cap = cv2.VideoCapture(str(top_video))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # 3. Run inference (2-timestep mode)
        print(f"  Running inference on {n_frames} frames...")
        results = evaluator.evaluate_video_2timesteps_advantages(
            video_paths=(str(top_video), str(left_video), str(right_video)),
            prompt=prompt,
            batch_size=batch_size,
            frame_interval=1,
            relative_interval=50,
            min_frame_index=0,
            max_frame_index=n_frames - 1,
        )

        # Keep absolute_value unchanged as progress. The evaluator directly predicts
        # model(frame_n, frame_{n+50}); align that result to frame n+50 so the
        # displayed value at frame k is model(frame_{k-50}, frame_k).
        progress_values, relative_advantages = build_overlay_values(results, n_frames, relative_interval=50)

        # 4. Generate overlay videos for each camera view
        for video_key, view_name in camera_views:
            src_video = dataset_dir / video_path_template.format(
                episode_chunk=chunk_idx, episode_index=ep_idx, video_key=video_key
            )
            dst_video = output_dir / f"episode_{ep_idx:06d}_{view_name}_overlay.mp4"
            process_video_with_overlay(
                src_video,
                dst_video,
                progress_values,
                relative_advantages,
                fps=fps,
                playback_speed=playback_speed,
                font_scale=font_scale,
                thickness=thickness,
                relative_threshold=relative_threshold,
                overlay_alpha=overlay_alpha,
            )

        # Also save raw predictions as JSON for reference
        pred_path = output_dir / f"episode_{ep_idx:06d}_predictions.json"
        with open(pred_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved predictions: {pred_path}")

    evaluator.shutdown()
    print(f"\n[OK] All done! Output videos saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config-name", type=str, default="ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD")
    parser.add_argument("--prompt", type=str, default="Flatten and fold the cloth.")
    parser.add_argument("--output-dir", type=Path, default=Path("./eval_overlay_output"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=0.5,
        help="Output playback speed multiplier; 0.5 doubles the video duration.",
    )
    parser.add_argument("--font-scale", type=float, default=0.6)
    parser.add_argument("--thickness", type=int, default=2)
    parser.add_argument(
        "--relative-threshold",
        "--diff-threshold",
        dest="relative_threshold",
        type=float,
        default=0.08,
        help="Relative-advantage threshold below which a frame is considered stalled.",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.3, help="Alpha transparency of the red stall overlay.")
    args = parser.parse_args()

    run_eval_and_overlay(
        ckpt_dir=args.ckpt_dir,
        dataset_dir=args.dataset,
        config_name=args.config_name,
        prompt=args.prompt,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        playback_speed=args.playback_speed,
        font_scale=args.font_scale,
        thickness=args.thickness,
        relative_threshold=args.relative_threshold,
        overlay_alpha=args.overlay_alpha,
    )


if __name__ == "__main__":
    main()
