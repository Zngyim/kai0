#!/usr/bin/env python3
"""Upload LeRobot v2.1 dataset to HuggingFace Hub with episodes_stats.jsonl and version tag.

This script solves three problems at once:
  1. Generate missing meta/episodes_stats.jsonl (required by LeRobot v2.1)
  2. Upload the dataset to HuggingFace Hub
  3. Create a version tag (e.g., v2.1) matching codebase_version in info.json

Usage:
    # Basic usage
    uv run python scripts/upload_lerobot_to_hf.py \
        --dataset-dir ./my_dataset \
        --repo-id username/my_dataset

    # With explicit token and private repo
    uv run python scripts/upload_lerobot_to_hf.py \
        --dataset-dir ./my_dataset \
        --repo-id username/my_dataset \
        --token hf_xxx \
        --private

    # Skip steps if needed
    uv run python scripts/upload_lerobot_to_hf.py \
        --dataset-dir ./my_dataset \
        --repo-id username/my_dataset \
        --skip-stats   # episodes_stats.jsonl already exists
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, upload_folder


def get_feature_stats(array: np.ndarray, axis: tuple, keepdims: bool) -> dict:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims).tolist(),
        "max": np.max(array, axis=axis, keepdims=keepdims).tolist(),
        "mean": np.mean(array, axis=axis, keepdims=keepdims).tolist(),
        "std": np.std(array, axis=axis, keepdims=keepdims).tolist(),
        "count": [len(array)],
    }


def generate_episodes_stats(dataset_dir: Path) -> Path:
    """Generate meta/episodes_stats.jsonl for a LeRobot v2.1 dataset."""
    meta_dir = dataset_dir / "meta"
    data_dir = dataset_dir / "data"
    output_file = meta_dir / "episodes_stats.jsonl"

    if not meta_dir.exists():
        raise FileNotFoundError(f"Meta directory not found: {meta_dir}")

    # Load info.json
    info_path = meta_dir / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    features = info["features"]
    data_path_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")

    # Load episodes.jsonl
    episodes_path = meta_dir / "episodes.jsonl"
    with open(episodes_path) as f:
        episodes = [json.loads(line) for line in f]

    # Remove old stats file if exists
    output_file.unlink(missing_ok=True)

    for ep in episodes:
        ep_idx = ep["episode_index"]
        chunk_idx = ep.get("chunk_index", 0)

        # Build parquet path from template
        parquet_name = f"episode_{ep_idx:06d}.parquet"
        parquet_path = data_dir / f"chunk-{chunk_idx:03d}" / parquet_name

        if not parquet_path.exists():
            # Fallback: try flat structure
            parquet_path = data_dir / "chunk-000" / parquet_name
            if not parquet_path.exists():
                raise FileNotFoundError(f"Parquet not found for episode {ep_idx}: {parquet_path}")

        df = pd.read_parquet(parquet_path)

        ep_stats = {}
        for key, ft in features.items():
            if ft["dtype"] in ("string", "image", "video"):
                continue
            if key not in df.columns:
                continue

            data = np.array(df[key].tolist())
            if data.ndim == 1 and len(df) > 0 and isinstance(df[key].iloc[0], (list, np.ndarray)):
                data = np.vstack(df[key].tolist())

            axes_to_reduce = 0
            keepdims = data.ndim == 1
            ep_stats[key] = get_feature_stats(data, axis=axes_to_reduce, keepdims=keepdims)

        record = {"episode_index": ep_idx, "stats": ep_stats}
        with open(output_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"  Episode {ep_idx}: stats for {len(ep_stats)} features")

    print(f"[OK] Generated: {output_file}")
    return output_file


def ensure_chunks_size_valid(dataset_dir: Path) -> None:
    """Check and warn if chunks_size may cause episode lookup issues."""
    info_path = dataset_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    total_episodes = info.get("total_episodes", 0)
    chunks_size = info.get("chunks_size", 1)

    if total_episodes > chunks_size:
        # This is normal for multi-chunk datasets
        pass
    else:
        # If all episodes fit in chunk-000, verify they are actually there
        chunk_000 = dataset_dir / "data" / "chunk-000"
        if chunk_000.exists():
            parquet_files = list(chunk_000.glob("episode_*.parquet"))
            if len(parquet_files) == total_episodes:
                print(f"[OK] All {total_episodes} episodes found in chunk-000 (chunks_size={chunks_size})")
            else:
                print(f"[WARN] chunk-000 has {len(parquet_files)} files but total_episodes={total_episodes}")


def upload_dataset(
    dataset_dir: Path,
    repo_id: str,
    token: str | None,
    private: bool = False,
) -> None:
    """Upload dataset folder to HuggingFace Hub."""
    print(f"[INFO] Uploading to HuggingFace: {repo_id} ...")

    # Ensure repo exists with correct visibility (upload_folder does not accept 'private')
    api = HfApi(token=token)
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            private=private,
            exist_ok=True,
        )
    except Exception as e:
        print(f"[WARN] Could not create/verify repo: {e}")

    upload_folder(
        repo_id=repo_id,
        folder_path=str(dataset_dir),
        repo_type="dataset",
        token=token,
        delete_patterns="*",  # Clean upload: remove old files
    )
    print(f"[OK] Upload complete: https://huggingface.co/datasets/{repo_id}")


def create_version_tag(
    repo_id: str,
    dataset_dir: Path,
    token: str | None,
    tag: str | None = None,
) -> None:
    """Create a version tag on the HF repo."""
    api = HfApi(token=token)

    # Auto-detect tag from info.json if not provided
    if tag is None:
        info_path = dataset_dir / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)
        tag = info.get("codebase_version", "v2.1")

    print(f"[INFO] Creating tag '{tag}' on {repo_id} ...")
    try:
        api.create_tag(repo_id, tag=tag, repo_type="dataset", exist_ok=True)
        print(f"[OK] Tag '{tag}' created (or already exists)")
    except Exception as e:
        print(f"[ERROR] Failed to create tag: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Upload LeRobot dataset to HuggingFace with stats and version tag."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to local LeRobot dataset root (contains data/, meta/, videos/)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID, e.g. 'username/my-dataset'",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token. Defaults to HF_TOKEN env var.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private if it doesn't exist.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Version tag to create. Defaults to codebase_version from info.json.",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip generating episodes_stats.jsonl (assume it already exists).",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip uploading to HuggingFace (only generate stats and tag locally).",
    )
    parser.add_argument(
        "--skip-tag",
        action="store_true",
        help="Skip creating version tag on HF repo.",
    )

    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    # Verify required files exist
    required_files = ["meta/info.json", "meta/episodes.jsonl", "data"]
    for rel_path in required_files:
        if not (dataset_dir / rel_path).exists():
            print(f"[ERROR] Required path missing: {dataset_dir / rel_path}")
            sys.exit(1)

    print(f"[INFO] Dataset: {dataset_dir}")
    print(f"[INFO] Target HF repo: {args.repo_id}")

    # Step 1: Generate episodes_stats.jsonl
    if not args.skip_stats:
        print("\n[STEP 1/3] Generating episodes_stats.jsonl ...")
        try:
            ensure_chunks_size_valid(dataset_dir)
            generate_episodes_stats(dataset_dir)
        except Exception as e:
            print(f"[ERROR] Failed to generate episodes_stats.jsonl: {e}")
            sys.exit(1)
    else:
        print("\n[STEP 1/3] Skipped: episodes_stats.jsonl generation")

    # Step 2: Upload to HuggingFace
    if not args.skip_upload:
        print("\n[STEP 2/3] Uploading dataset to HuggingFace ...")
        try:
            upload_dataset(
                dataset_dir=dataset_dir,
                repo_id=args.repo_id,
                token=args.token,
                private=args.private,
            )
        except Exception as e:
            print(f"[ERROR] Upload failed: {e}")
            sys.exit(1)
    else:
        print("\n[STEP 2/3] Skipped: HuggingFace upload")

    # Step 3: Create version tag
    if not args.skip_tag:
        if args.skip_upload:
            print("\n[STEP 3/3] Skipped: cannot create tag without upload")
        else:
            print("\n[STEP 3/3] Creating version tag ...")
            try:
                create_version_tag(
                    repo_id=args.repo_id,
                    dataset_dir=dataset_dir,
                    token=args.token,
                    tag=args.tag,
                )
            except Exception as e:
                print(f"[ERROR] Failed to create tag: {e}")
                sys.exit(1)
    else:
        print("\n[STEP 3/3] Skipped: version tag creation")

    print("\n[ALL DONE]")
    print(f"  Dataset: {args.repo_id}")
    if not args.skip_tag and not args.skip_upload:
        tag = args.tag or "(auto-detected from info.json)"
        print(f"  Tag: {tag}")
    print(f"  Local path: {dataset_dir}")


if __name__ == "__main__":
    main()
