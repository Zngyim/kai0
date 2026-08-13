#!/usr/bin/env python3
"""Run Step 2 eval on a dataset with checkpoint, compute eval loss, and visualize results.

Usage:
    cd /mnt/pfs/zhangjiyao/yiming/kai0
    uv run python scripts/eval_and_visualize.py \
        --ckpt-dir /mnt/pfs/zhangjiyao/yiming/checkpoints/STACK_BLOCKS_ADVANTAGE/run2/10000 \
        --dataset  /mnt/pfs/zhangjiyao/yiming/kai0/testdata/test_data-525_lerobot \
        --config-name STACK_BLOCKS_ADVANTAGE \
        --prompt "Stack three blocks." \
        --output-dir ./eval_viz0710_disrupt_mtrain_neval
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stage_advantage.annotation.evaluator import SimpleValueEvaluator
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata


def run_eval_and_visualize(
    ckpt_dir: Path,
    dataset_dir: Path,
    config_name: str,
    prompt: str,
    output_dir: Path,
    batch_size: int = 8,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load evaluator
    print(f"[INFO] Loading checkpoint from: {ckpt_dir}")
    evaluator = SimpleValueEvaluator(
        config_name=config_name,
        ckpt_dir=str(ckpt_dir),
        num_workers=4,
    )

    # 2. Load dataset metadata
    dataset_meta = LeRobotDatasetMetadata(str(dataset_dir))
    total_episodes = dataset_meta.total_episodes
    print(f"[INFO] Dataset: {dataset_dir}, Total episodes: {total_episodes}")

    # 3. Run eval episode by episode
    all_results = []
    for ep_idx in range(total_episodes):
        print(f"[INFO] Processing episode {ep_idx} ...")

        # Build video paths
        chunk_idx = ep_idx // dataset_meta.chunks_size
        video_dir = dataset_dir / "videos" / f"chunk-{chunk_idx:03d}"
        top_video = video_dir / "observation.images.top_head" / f"episode_{ep_idx:06d}.mp4"
        left_video = video_dir / "observation.images.hand_left" / f"episode_{ep_idx:06d}.mp4"
        right_video = video_dir / "observation.images.hand_right" / f"episode_{ep_idx:06d}.mp4"

        if not top_video.exists() or not left_video.exists() or not right_video.exists():
            print(f"  [WARN] Missing video for episode {ep_idx}, skipping")
            continue

        # Read original parquet for ground truth
        parquet_path = (
            dataset_dir
            / "data"
            / f"chunk-{chunk_idx:03d}"
            / f"episode_{ep_idx:06d}.parquet"
        )
        df_orig = pd.read_parquet(parquet_path)
        n_frames = len(df_orig)

        # Run inference
        results = evaluator.evaluate_video_2timesteps_advantages(
            video_paths=(str(top_video), str(left_video), str(right_video)),
            prompt=prompt,
            batch_size=batch_size,
            frame_interval=1,
            relative_interval=50,
            min_frame_index=0,
            max_frame_index=n_frames - 1,
        )

        # Build result dataframe
        pred_values = np.full(n_frames, np.nan, dtype=np.float32)
        pred_abs_adv = np.full(n_frames, np.nan, dtype=np.float32)
        pred_rel_adv = np.full(n_frames, np.nan, dtype=np.float32)

        for r in results:
            fidx = r["frame_idx"]
            if 0 <= fidx < n_frames:
                pred_values[fidx] = r.get("absolute_value", np.nan)
                pred_abs_adv[fidx] = r.get("absolute_advantage", np.nan)
                pred_rel_adv[fidx] = r.get("relative_advantage", np.nan)

        df_out = df_orig.copy()
        df_out["absolute_value"] = pred_values
        df_out["absolute_advantage"] = pred_abs_adv
        df_out["relative_advantage"] = pred_rel_adv

        # Save annotated parquet
        out_parquet_dir = output_dir / "data" / f"chunk-{chunk_idx:03d}"
        out_parquet_dir.mkdir(parents=True, exist_ok=True)
        out_parquet_path = out_parquet_dir / f"episode_{ep_idx:06d}.parquet"
        df_out.to_parquet(out_parquet_path, index=False)

        # 4. Compute eval metrics
        gt = df_out["stage_progress_gt"].values
        valid_mask = ~np.isnan(pred_values)
        if valid_mask.sum() > 0:
            mse = np.mean((pred_values[valid_mask] - gt[valid_mask]) ** 2)
            corr = np.corrcoef(pred_values[valid_mask], gt[valid_mask])[0, 1]
        else:
            mse = np.nan
            corr = np.nan

        print(f"  Episode {ep_idx}: MSE={mse:.6f}, Corr={corr:.4f}, frames={valid_mask.sum()}")

        # 5. Visualize
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        axes[0].plot(gt, label="stage_progress_gt", color="blue")
        axes[0].set_ylabel("Progress GT")
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_title(f"Episode {ep_idx} | MSE={mse:.6f} | Corr={corr:.4f}")

        axes[1].plot(pred_values, label="absolute_value (pred)", color="green")
        axes[1].set_ylabel("Absolute Value")
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(pred_abs_adv, label="absolute_advantage", color="red", alpha=0.7)
        axes[2].plot(pred_rel_adv, label="relative_advantage", color="orange", alpha=0.7)
        axes[2].set_ylabel("Advantage")
        axes[2].set_xlabel("Frame")
        axes[2].legend()
        axes[2].grid(True)

        viz_path = output_dir / f"episode_{ep_idx:06d}_advantage.png"
        plt.savefig(viz_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        all_results.append({
            "episode_index": ep_idx,
            "mse": mse,
            "correlation": corr,
            "valid_frames": int(valid_mask.sum()),
        })

    evaluator.shutdown()

    # 6. Summary
    summary_df = pd.DataFrame(all_results)
    summary_path = output_dir / "eval_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[OK] Done! Summary saved to: {summary_path}")
    print(summary_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config-name", type=str, default="ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD")
    parser.add_argument("--prompt", type=str, default="Flatten and fold the cloth.")
    parser.add_argument("--output-dir", type=Path, default=Path("./eval_viz"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    run_eval_and_visualize(
        ckpt_dir=args.ckpt_dir,
        dataset_dir=args.dataset,
        config_name=args.config_name,
        prompt=args.prompt,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
