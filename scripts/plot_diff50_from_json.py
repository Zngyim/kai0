#!/usr/bin/env python3
"""Plot diff50 curve from existing predictions JSON.

    uv run python scripts/plot_diff50_from_json.py \
        --predictions ./test_data_sa_overlay/episode_000000_predictions.json \
        --output ./test_data_sa_overlay/diff50_curve_ep000000.png

"""



import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("./diff50_curve.png"))
    args = parser.parse_args()

    with open(args.predictions, "r") as f:
        results = json.load(f)

    n_frames = max(r["frame_idx"] for r in results) + 1
    abs_values = np.full(n_frames, np.nan, dtype=np.float32)
    rel_advantages = np.full(n_frames, np.nan, dtype=np.float32)
    abs_advantages = np.full(n_frames, np.nan, dtype=np.float32)

    for r in results:
        fidx = r["frame_idx"]
        if 0 <= fidx < n_frames:
            abs_values[fidx] = r.get("absolute_value", np.nan)
            rel_advantages[fidx] = r.get("relative_advantage", np.nan)
            abs_advantages[fidx] = r.get("absolute_advantage", np.nan)

    diff50 = np.full(n_frames, np.nan, dtype=np.float32)
    for i in range(n_frames):
        if not np.isnan(abs_values[i]):
            prev = max(0, i - 50)
            diff50[i] = abs_values[i] - abs_values[prev]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    if n_frames > 50:
        for i, ax in enumerate(axes):
            ax.axvline(
                50,
                color="purple",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                zorder=10,
                label="frame 50" if i == 0 else None,
            )

    axes[0].plot(abs_values, label="absolute_value", color="blue", linewidth=1.5)
    axes[0].set_ylabel("Absolute Value")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Advantage Prediction Curves")

    axes[1].plot(diff50, label="diff50 = current - 50 frames ago", color="red", linewidth=1.5)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1].set_ylabel("Diff 50")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(abs_advantages, label="absolute_advantage (from model)", color="green", linewidth=1.5)
    axes[2].plot(rel_advantages, label="relative_advantage (from model)", color="orange", linewidth=1.5, alpha=0.7)
    axes[2].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[2].set_ylabel("Model Advantage")
    axes[2].set_xlabel("Frame")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"[OK] Plot saved to: {args.output}")


if __name__ == "__main__":
    main()
