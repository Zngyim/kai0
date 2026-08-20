#!/usr/bin/env python3
"""Annotate or re-annotate stage_progress_gt on an existing LeRobot v2.1 dataset.

The labels can either be generated from augmentation metadata (perturbation /
recovery intervals plus stage boundaries) or linearly from 0 to 1 over every
episode. The script updates both the parquet files and the LeRobot metadata.

Usage:
    python annotate_stage_progress_gt.py \
        --dataset-dir ./test_data-525_lerobot \
        --aug-meta ./test_data-525/augmentation_metadata.json \
        --k-stages 3

    python annotate_stage_progress_gt.py \
        --dataset-dir ./my_lerobot_dataset \
        --linear \
        --exclude-feature extra_view_image-0

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
import pyarrow as pa
import pyarrow.parquet as pq


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


def compute_linear_progress_gt(n_frames: int) -> np.ndarray:
    """Return frame-wise progress that spans exactly [0, 1]."""
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    if n_frames == 1:
        return np.zeros(1, dtype=np.float32)
    return np.linspace(0.0, 1.0, num=n_frames, dtype=np.float32)


def compute_scalar_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    """Compute LeRobot-compatible scalar statistics."""
    return {
        "min": [float(np.min(values))],
        "max": [float(np.max(values))],
        "mean": [float(np.mean(values))],
        "std": [float(np.std(values))],
        "count": [len(values)],
        "q01": [float(np.quantile(values, 0.01))],
        "q10": [float(np.quantile(values, 0.10))],
        "q50": [float(np.quantile(values, 0.50))],
        "q90": [float(np.quantile(values, 0.90))],
        "q99": [float(np.quantile(values, 0.99))],
    }


def update_metadata(
    dataset_dir: Path,
    labels_by_episode: dict[int, np.ndarray],
    excluded_features: list[str],
) -> None:
    """Keep info.json and stats files consistent with the annotated parquet files."""
    meta_dir = dataset_dir / "meta"
    info_path = meta_dir / "info.json"
    with info_path.open() as f:
        info = json.load(f)

    features = info.setdefault("features", {})
    for feature in excluded_features:
        removed = features.pop(feature, None)
        if removed is not None and removed.get("dtype") == "video":
            info["total_videos"] = max(0, int(info.get("total_videos", 0)) - 1)
    features["stage_progress_gt"] = {
        "dtype": "float32",
        "shape": [1],
        "names": ["stage_progress_gt"],
    }
    info_path.write_text(json.dumps(info, indent=2) + "\n")

    episodes_stats_path = meta_dir / "episodes_stats.jsonl"
    if episodes_stats_path.exists():
        records = [json.loads(line) for line in episodes_stats_path.read_text().splitlines() if line.strip()]
        for record in records:
            episode_index = int(record["episode_index"])
            if episode_index in labels_by_episode:
                stats = record.setdefault("stats", {})
                for feature in excluded_features:
                    stats.pop(feature, None)
                stats["stage_progress_gt"] = compute_scalar_stats(labels_by_episode[episode_index])
        episodes_stats_path.write_text("".join(json.dumps(record) + "\n" for record in records))

    all_labels = np.concatenate([labels_by_episode[index] for index in sorted(labels_by_episode)])
    stats_path = meta_dir / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    for feature in excluded_features:
        stats.pop(feature, None)
    stats["stage_progress_gt"] = compute_scalar_stats(all_labels)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Annotate stage_progress_gt on a LeRobot dataset.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to LeRobot dataset root")
    label_source = parser.add_mutually_exclusive_group(required=True)
    label_source.add_argument("--aug-meta", type=Path, help="Path to augmentation_metadata.json")
    label_source.add_argument(
        "--linear",
        action="store_true",
        help="Linearly label every episode from 0 at its first frame to 1 at its last frame",
    )
    parser.add_argument("--k-stages", type=int, default=3, help="Number of stages")
    parser.add_argument(
        "--exclude-feature",
        action="append",
        default=[],
        help="Remove a duplicate or unused feature from LeRobot metadata (repeatable)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if args.aug_meta is not None:
        with args.aug_meta.open() as f:
            aug_meta = json.load(f)
    else:
        aug_meta = {}

    # Discover parquet files
    data_dir = dataset_dir / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    labels_by_episode: dict[int, np.ndarray] = {}
    for parquet_path in parquet_files:
        # Extract episode index from filename (episode_000000.parquet -> 0)
        ep_idx = int(parquet_path.stem.split("_")[1])
        ep_key = str(ep_idx)

        table = pq.read_table(parquet_path)
        n_frames = table.num_rows

        if not args.linear and ep_key not in aug_meta:
            print(f"Skipping episode {ep_idx}: not found in augmentation metadata")
            continue

        if args.linear:
            subtask_indices = []
            new_gt = compute_linear_progress_gt(n_frames)
        else:
            ep_meta = aug_meta[ep_key]
            subtask_indices = ep_meta.get("subtask_completion_indices", [])
            segments = ep_meta.get("segments", [])
            new_gt = compute_stage_progress_gt(
                n_frames=n_frames,
                subtask_completion_indices=subtask_indices,
                segments=segments,
                k_stages=args.k_stages,
            )
        labels_by_episode[ep_idx] = new_gt

        if args.dry_run:
            print(f"Episode {ep_idx}: n_frames={n_frames}, gt_range=[{new_gt.min():.4f}, {new_gt.max():.4f}]")
            if not args.linear:
                boundaries = [0] + subtask_indices[: args.k_stages - 1] + [n_frames]
                for stage_k in range(args.k_stages):
                    s, e = boundaries[stage_k], boundaries[stage_k + 1]
                    print(
                        f"  Stage {stage_k}: [{s}, {e}) -> gt_range=[{new_gt[s:e].min():.4f}, {new_gt[s:e].max():.4f}]"
                    )
            continue

        if "stage_progress_gt" in table.column_names:
            table = table.drop_columns(["stage_progress_gt"])

        gt_array = pa.array(new_gt, type=pa.float32())
        table = table.append_column("stage_progress_gt", gt_array)
        temporary_path = parquet_path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary_path)
        temporary_path.replace(parquet_path)
        print(f"Updated episode {ep_idx}: n_frames={n_frames}, gt_range=[{new_gt.min():.4f}, {new_gt.max():.4f}]")

    if not args.dry_run:
        update_metadata(dataset_dir, labels_by_episode, args.exclude_feature)

    print("\nDone!")


if __name__ == "__main__":
    main()
