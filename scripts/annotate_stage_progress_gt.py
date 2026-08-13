#!/usr/bin/env python3
"""Annotate or re-annotate stage_progress_gt on an existing LeRobot v2.1 dataset.

This script reads augmentation_metadata.json (perturbation/recovery intervals +
stage boundaries) and updates the `stage_progress_gt` column in each episode's
parquet file.

Usage:
    python annotate_stage_progress_gt.py \
        --dataset-dir ./test_data-525_lerobot \
        --aug-meta ./test_data-525/augmentation_metadata.json \
        --k-stages 3

The labeling strategy:
  - Episode is divided into K stages using subtask_completion_indices[:K-1].
    The last stage extends to the final frame.
  - Inside each stage:
    - Perturbation frames: accumulate -1
    - Normal / recovery frames: accumulate +1
  - Each stage is then linearly scaled to [k/K, (k+1)/K].
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa


def compute_stage_progress_gt(n_frames: int, subtask_completion_indices: list, segments: list, k_stages: int = 3):
    """Compute stage_progress_gt with perturbation-aware logic."""
    boundaries = [0] + subtask_completion_indices[: k_stages - 1] + [n_frames]

    perturb_mask = np.zeros(n_frames, dtype=bool)
    recovery_mask = np.zeros(n_frames, dtype=bool)
    for seg in segments:
        ps, pe = seg["perturb_start_hdf5"], seg["perturb_end_hdf5"]
        rs, re = seg["recovery_start_hdf5"], seg["recovery_end_hdf5"]
        perturb_mask[ps:pe] = True
        recovery_mask[rs:re] = True

    stage_progress_gt = np.zeros(n_frames, dtype=np.float32)

    for stage_k in range(k_stages):
        start = boundaries[stage_k]
        end = boundaries[stage_k + 1]
        stage_len = end - start
        if stage_len <= 0:
            continue

        stage_gt = np.zeros(stage_len, dtype=np.float32)
        current = 0.0
        for i in range(stage_len):
            frame_idx = start + i
            if perturb_mask[frame_idx]:
                current -= 1.0
            else:
                current += 1.0
            stage_gt[i] = current

        min_val = stage_gt.min()
        max_val = stage_gt.max()
        stage_low = stage_k / k_stages
        stage_high = (stage_k + 1) / k_stages

        if max_val > min_val:
            stage_gt = (stage_gt - min_val) / (max_val - min_val) * (stage_high - stage_low) + stage_low
        else:
            stage_gt = np.full(stage_len, (stage_low + stage_high) / 2.0, dtype=np.float32)

        stage_progress_gt[start:end] = stage_gt

    return stage_progress_gt


def main():
    parser = argparse.ArgumentParser(description="Annotate stage_progress_gt on a LeRobot dataset.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to LeRobot dataset root")
    parser.add_argument("--aug-meta", type=Path, required=True, help="Path to augmentation_metadata.json")
    parser.add_argument("--k-stages", type=int, default=3, help="Number of stages")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    aug_meta_path = args.aug_meta

    with open(aug_meta_path, "r") as f:
        aug_meta = json.load(f)

    # Discover parquet files
    data_dir = dataset_dir / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    for parquet_path in parquet_files:
        # Extract episode index from filename (episode_000000.parquet -> 0)
        ep_idx = int(parquet_path.stem.split("_")[1])
        ep_key = str(ep_idx)

        df = pd.read_parquet(parquet_path)
        n_frames = len(df)

        if ep_key not in aug_meta:
            print(f"Skipping episode {ep_idx}: not found in augmentation metadata")
            continue

        ep_meta = aug_meta[ep_key]
        subtask_indices = ep_meta.get("subtask_completion_indices", [])
        segments = ep_meta.get("segments", [])

        new_gt = compute_stage_progress_gt(
            n_frames=n_frames,
            subtask_completion_indices=subtask_indices,
            segments=segments,
            k_stages=args.k_stages,
        )

        if args.dry_run:
            print(f"Episode {ep_idx}: n_frames={n_frames}, gt_range=[{new_gt.min():.4f}, {new_gt.max():.4f}]")
            # Print per-stage stats
            boundaries = [0] + subtask_indices[: args.k_stages - 1] + [n_frames]
            for stage_k in range(args.k_stages):
                s, e = boundaries[stage_k], boundaries[stage_k + 1]
                print(f"  Stage {stage_k}: [{s}, {e}) -> gt_range=[{new_gt[s:e].min():.4f}, {new_gt[s:e].max():.4f}]")
            continue

        # Update parquet
        table = pq.read_table(parquet_path)
        if "stage_progress_gt" in table.column_names:
            table = table.drop_columns(["stage_progress_gt"])

        gt_array = pa.array(new_gt.tolist(), type=pa.float32())
        table = table.append_column("stage_progress_gt", gt_array)
        pq.write_table(table, parquet_path)
        print(f"Updated episode {ep_idx}: n_frames={n_frames}, gt_range=[{new_gt.min():.4f}, {new_gt.max():.4f}]")

    print("\nDone!")


if __name__ == "__main__":
    main()
