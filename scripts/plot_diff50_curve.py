#!/usr/bin/env python3
"""Quickly run advantage inference and plot the diff50 curve."""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stage_advantage.annotation.evaluator import SimpleValueEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("--top-video", type=Path, required=True)
    parser.add_argument("--left-video", type=Path, required=True)
    parser.add_argument("--right-video", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="Stack three blocks.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("./diff50_curve.png"))
    args = parser.parse_args()

    print(f"[INFO] Loading checkpoint from: {args.ckpt_dir}")
    evaluator = SimpleValueEvaluator(
        config_name=args.config_name,
        ckpt_dir=str(args.ckpt_dir),
        num_workers=4,
    )

    print("[INFO] Running inference...")
    results = evaluator.evaluate_video_2timesteps_advantages(
        video_paths=(str(args.top_video), str(args.left_video), str(args.right_video)),
        prompt=args.prompt,
        batch_size=args.batch_size,
        frame_interval=1,
        relative_interval=50,
        min_frame_index=0,
        max_frame_index=None,
    )
    evaluator.shutdown()

    n_frames = max(r["frame_idx"] for r in results) + 1
    abs_values = np.full(n_frames, np.nan, dtype=np.float32)
    for r in results:
        fidx = r["frame_idx"]
        if 0 <= fidx < n_frames:
            abs_values[fidx] = r.get("absolute_value", np.nan)

    diff50 = np.full(n_frames, np.nan, dtype=np.float32)
    for i in range(n_frames):
        if not np.isnan(abs_values[i]):
            prev = max(0, i - 50)
            diff50[i] = abs_values[i] - abs_values[prev]

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(abs_values, label="absolute_value", color="blue")
    axes[0].set_ylabel("Absolute Value")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title("Advantage Prediction")

    axes[1].plot(diff50, label="diff50 (current - 50 frames ago)", color="red")
    axes[1].set_ylabel("Diff 50")
    axes[1].set_xlabel("Frame")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"[OK] Plot saved to: {args.output}")


if __name__ == "__main__":
    main()
