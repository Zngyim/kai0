#!/usr/bin/env python3
"""Generate missing meta/episodes_stats.jsonl for LeRobot v2.1 dataset.

Supports multi-chunk datasets. Automatically locates the correct chunk directory
for each episode based on episodes.jsonl::chunk_index.

Usage:
    uv run python ./scripts/generate_episodes_stats.py --dataset-root /mnt/pfs/zhangjiyao/yiming/kai0/testdata/0710_disrupt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def get_feature_stats(array: np.ndarray, axis: tuple, keepdims: bool) -> dict:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims).tolist(),
        "max": np.max(array, axis=axis, keepdims=keepdims).tolist(),
        "mean": np.mean(array, axis=axis, keepdims=keepdims).tolist(),
        "std": np.std(array, axis=axis, keepdims=keepdims).tolist(),
        "count": [len(array)],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate meta/episodes_stats.jsonl for a LeRobot v2.1 dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the LeRobot dataset root (must contain meta/ and data/)",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    meta_dir = dataset_root / "meta"
    data_root = dataset_root / "data"
    output_file = meta_dir / "episodes_stats.jsonl"

    if not meta_dir.exists():
        raise FileNotFoundError(f"Meta directory not found: {meta_dir}")
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")

    # Load info.json to get feature specs
    info_path = meta_dir / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found: {info_path}")
    with open(info_path) as f:
        info = json.load(f)
    features = info["features"]

    # Load episodes.jsonl
    episodes_path = meta_dir / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found: {episodes_path}")
    with open(episodes_path) as f:
        episodes = [json.loads(line) for line in f]

    output_file.unlink(missing_ok=True)

    missing_parquets = []
    for ep in episodes:
        ep_idx = ep["episode_index"]
        chunk_idx = ep.get("chunk_index", 0)
        data_dir = data_root / f"chunk-{chunk_idx:03d}"
        parquet_path = data_dir / f"episode_{ep_idx:06d}.parquet"

        if not parquet_path.exists():
            missing_parquets.append(str(parquet_path))
            print(f"  Warning: {parquet_path} not found, skipping episode {ep_idx}")
            continue

        df = pd.read_parquet(parquet_path)

        ep_stats = {}
        for key, ft in features.items():
            if ft["dtype"] == "string":
                continue
            elif ft["dtype"] in ["image", "video"]:
                # Skip video stats (not used in normalization for advantage estimator)
                continue
            elif key not in df.columns:
                continue
            else:
                # Numerical features
                data = np.array(df[key].tolist())
                if data.ndim == 1 and isinstance(df[key].iloc[0], (list, np.ndarray)):
                    # Handle list-like columns (e.g., observation.state stored as list in parquet)
                    data = np.vstack(df[key].tolist())

                axes_to_reduce = 0
                keepdims = data.ndim == 1
                ep_stats[key] = get_feature_stats(data, axis=axes_to_reduce, keepdims=keepdims)

        # Write in LeRobot format
        record = {"episode_index": ep_idx, "stats": ep_stats}
        with open(output_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"Episode {ep_idx} (chunk {chunk_idx}): wrote stats for {list(ep_stats.keys())}")

    print(f"\nDone: {output_file}")
    print(f"Total episodes processed: {len(episodes) - len(missing_parquets)} / {len(episodes)}")
    if missing_parquets:
        print(f"Missing parquet files: {len(missing_parquets)}")


if __name__ == "__main__":
    main()
