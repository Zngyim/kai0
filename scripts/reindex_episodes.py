#!/usr/bin/env python3
"""Reindex LeRobot v2.1 dataset episodes to start from 0 with contiguous indices.

This script:
  1. Reads meta/episodes.jsonl and sorts by start_index (preserves temporal order)
  2. Reassigns episode_index as 0, 1, 2, ...
  3. Updates parquet files (internal episode_index column + file rename)
  4. Renames video files accordingly
  5. Updates episodes.jsonl, info.json (splits), and episodes_stats.jsonl

Usage:
    # Dry-run to preview changes
    uv run python scripts/reindex_episodes.py --dataset-root ./my_dataset --dry-run

    # Execute in-place
    uv run python scripts/reindex_episodes.py --dataset-root ./my_dataset
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def build_rename_plan(dataset_dir: Path) -> tuple[list[dict], dict[int, int]]:
    """Build the renaming plan without modifying anything.

    Returns:
        new_episodes: list of episode records with re-indexed episode_index
        old_to_new: mapping from old episode_index to new episode_index
    """
    meta_dir = dataset_dir / "meta"
    data_dir = dataset_dir / "data"
    video_dir = dataset_dir / "videos"

    episodes = load_jsonl(meta_dir / "episodes.jsonl")
    # Sort by start_index to preserve temporal order
    episodes.sort(key=lambda e: e["start_index"])

    with open(meta_dir / "info.json") as f:
        info = json.load(f)
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]

    new_episodes = []
    old_to_new = {}

    for new_idx, ep in enumerate(episodes):
        old_idx = ep["episode_index"]
        old_to_new[old_idx] = new_idx

        chunk_idx = ep.get("chunk_index", 0)

        # Parquet paths
        old_parquet = data_dir / f"chunk-{chunk_idx:03d}" / f"episode_{old_idx:06d}.parquet"
        new_parquet = data_dir / f"chunk-{chunk_idx:03d}" / f"episode_{new_idx:06d}.parquet"

        # Video paths
        video_moves = []
        for vk in video_keys:
            old_video = video_dir / f"chunk-{chunk_idx:03d}" / vk / f"episode_{old_idx:06d}.mp4"
            new_video = video_dir / f"chunk-{chunk_idx:03d}" / vk / f"episode_{new_idx:06d}.mp4"
            video_moves.append((old_video, new_video))

        new_ep = dict(ep)
        new_ep["episode_index"] = new_idx
        new_episodes.append({
            "record": new_ep,
            "chunk_index": chunk_idx,
            "old_parquet": old_parquet,
            "new_parquet": new_parquet,
            "video_moves": video_moves,
        })

    return new_episodes, old_to_new


def validate_plan(plan: list[dict], dataset_dir: Path) -> bool:
    """Check that all source files exist and there are no destination conflicts."""
    meta_dir = dataset_dir / "meta"
    ok = True

    # Check source files exist
    for item in plan:
        old_pq = item["old_parquet"]
        if not old_pq.exists():
            print(f"[ERROR] Source parquet missing: {old_pq}")
            ok = False

        for old_v, _ in item["video_moves"]:
            if not old_v.exists():
                print(f"[WARN] Source video missing: {old_v}")
                # Video missing is a warning, not fatal (some cameras may be absent)

    # Check destination conflicts
    dest_parquets = [item["new_parquet"] for item in plan]
    if len(dest_parquets) != len(set(dest_parquets)):
        print("[ERROR] Parquet destination conflict detected!")
        ok = False

    dest_videos = []
    for item in plan:
        for _, new_v in item["video_moves"]:
            dest_videos.append(str(new_v))
    if len(dest_videos) != len(set(dest_videos)):
        print("[ERROR] Video destination conflict detected!")
        ok = False

    # Warn if any destination already exists (shouldn't happen with contiguous 0-based indexing
    # unless the dataset already has episode_000000 etc.)
    for item in plan:
        if item["new_parquet"].exists() and item["new_parquet"] != item["old_parquet"]:
            print(f"[WARN] Destination parquet already exists: {item['new_parquet']}")
        for _, new_v in item["video_moves"]:
            if new_v.exists() and new_v not in [ov for ov, _ in item["video_moves"]]:
                print(f"[WARN] Destination video already exists: {new_v}")

    return ok


def preview_plan(plan: list[dict]) -> None:
    print("\n[PREVIEW] Planned changes:")
    for item in plan:
        old_idx = item["record"]["episode_index"]
        # old_idx is already new here... let's recover from filenames
        old_name = item["old_parquet"].name
        new_name = item["new_parquet"].name
        print(f"  {old_name} -> {new_name}")
        for ov, nv in item["video_moves"]:
            if ov.exists():
                print(f"    video: {ov.name} -> {nv.name}")
    print(f"\nTotal episodes to reindex: {len(plan)}\n")


def execute_plan(plan: list[dict], dataset_dir: Path, dry_run: bool) -> None:
    """Execute the renaming plan with two-phase safe strategy to avoid filename collisions."""
    meta_dir = dataset_dir / "meta"

    if dry_run:
        preview_plan(plan)
        return

    # Phase 0: Build old->new mapping for stats update
    old_to_new = {}
    for item in plan:
        old_idx = int(item["old_parquet"].stem.split("_")[1])
        new_idx = item["record"]["episode_index"]
        old_to_new[old_idx] = new_idx

    # Phase 1: Move all old files to .tmp to free up destination filenames
    print("[PHASE 1/3] Moving existing files to .tmp ...")
    tmp_parquets = []
    tmp_videos = []

    for item in plan:
        old_pq = item["old_parquet"]
        tmp_pq = old_pq.with_suffix(".parquet.tmp")
        if old_pq.exists() and old_pq != item["new_parquet"]:
            shutil.move(str(old_pq), str(tmp_pq))
            tmp_parquets.append((tmp_pq, item["new_parquet"], item["record"]["episode_index"]))
        elif old_pq.exists() and old_pq == item["new_parquet"]:
            # No rename needed, but still need to update internal episode_index
            tmp_parquets.append((old_pq, old_pq, item["record"]["episode_index"]))

        for old_v, new_v in item["video_moves"]:
            if old_v.exists() and old_v != new_v:
                tmp_v = old_v.with_suffix(".mp4.tmp")
                shutil.move(str(old_v), str(tmp_v))
                tmp_videos.append((tmp_v, new_v))
            elif old_v.exists() and old_v == new_v:
                tmp_videos.append((old_v, old_v))

    # Phase 2: Write new parquet files and rename videos from .tmp
    print("[PHASE 2/3] Writing new parquet files and renaming videos ...")
    for tmp_pq, new_pq, new_ep_idx in tmp_parquets:
        df = pd.read_parquet(tmp_pq)
        df["episode_index"] = new_ep_idx
        df.to_parquet(new_pq, index=False)

        if new_pq.exists() and tmp_pq != new_pq:
            tmp_pq.unlink()
        elif not new_pq.exists():
            print(f"[ERROR] Failed to write {new_pq}, aborting.")
            sys.exit(1)

        print(f"  {tmp_pq.name} -> {new_pq.name}")

    for tmp_v, new_v in tmp_videos:
        if tmp_v != new_v:
            shutil.move(str(tmp_v), str(new_v))
            print(f"  {tmp_v.name} -> {new_v.name}")

    # Phase 3: Update metadata files
    print("[PHASE 3/3] Updating metadata ...")
    new_episode_records = [item["record"] for item in plan]
    save_jsonl(meta_dir / "episodes.jsonl", new_episode_records)

    info_path = meta_dir / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    n_eps = len(plan)
    info["splits"] = {"train": f"0:{n_eps}"}
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    stats_path = meta_dir / "episodes_stats.jsonl"
    if stats_path.exists():
        old_stats = load_jsonl(stats_path)
        new_stats = []
        for stat in old_stats:
            old_ep_idx = stat["episode_index"]
            if old_ep_idx in old_to_new:
                new_stat = dict(stat)
                new_stat["episode_index"] = old_to_new[old_ep_idx]
                new_stats.append(new_stat)
        new_stats.sort(key=lambda s: s["episode_index"])
        save_jsonl(stats_path, new_stats)
        print(f"  Updated {stats_path}")

    print("\n[OK] Reindex complete!")
    print(f"  Total episodes: {n_eps}")
    print(f"  Episode indices now: 0 .. {n_eps - 1}")


def main():
    parser = argparse.ArgumentParser(
        description="Reindex LeRobot dataset episodes to contiguous 0-based indices."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the LeRobot dataset root (contains data/, meta/, videos/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying any files.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_root.resolve()
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    # Verify required paths
    required = ["meta/info.json", "meta/episodes.jsonl", "data", "videos"]
    for rel in required:
        if not (dataset_dir / rel).exists():
            print(f"[ERROR] Required path missing: {dataset_dir / rel}")
            sys.exit(1)

    print(f"[INFO] Dataset: {dataset_dir}")
    print(f"[INFO] Mode: {'dry-run' if args.dry_run else 'in-place'}")

    # Build plan
    plan, old_to_new = build_rename_plan(dataset_dir)

    # Validate
    print("[INFO] Validating plan ...")
    if not validate_plan(plan, dataset_dir):
        print("[ERROR] Validation failed. Aborting.")
        sys.exit(1)

    # Execute
    execute_plan(plan, dataset_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
