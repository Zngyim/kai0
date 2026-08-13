#!/usr/bin/env python3
"""Convert test_data HDF5 + debug_video to LeRobot v2.1 format with stage-aware stage_progress_gt.

This script processes datasets like `test_data-525` which contain:
  - data/episode*.hdf5
  - debug_video/episode*_<camera>.mp4
  - augmentation_metadata.json (stage boundaries + perturb/recovery segments)

The resulting LeRobot dataset can be used directly for Advantage Estimator training.
"""

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def compute_stage_progress_gt(n_frames: int, subtask_completion_indices: list, segments: list, k_stages: int = 3):
    """Compute stage_progress_gt with perturbation-aware logic.

    Rules:
      - Divide episode into K stages using subtask_completion_indices[:K-1].
        Stage 3 extends to the last frame.
      - Within each stage:
        - Normal / recovery frames: accumulate +1
        - Perturbation frames: accumulate -1
      - After accumulating, linearly scale each stage to [k/K, (k+1)/K].

    Returns:
        stage_progress_gt: np.ndarray of shape (n_frames,), dtype float32
    """
    # Build stage boundaries: [0, idx0, idx1, n_frames]
    boundaries = [0] + subtask_completion_indices[: k_stages - 1] + [n_frames]

    # Build perturb/recovery masks
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
                # Normal execution or recovery -> +1
                current += 1.0
            stage_gt[i] = current

        # Scale to [k/K, (k+1)/K]
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
    parser = argparse.ArgumentParser(description="Convert HDF5 test data to LeRobot v2.1 format.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Path to input dataset dir (e.g., test_data-525)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Path to output LeRobot dataset dir")
    parser.add_argument("--video-dir", type=Path, default=None, help="Directory containing debug videos (default: <input-dir>/debug_video)")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate")
    parser.add_argument("--video-height", type=int, default=240)
    parser.add_argument("--video-width", type=int, default=320)
    parser.add_argument("--video-codec", type=str, default="h264")
    args = parser.parse_args()

    src_data_dir = args.input_dir / "data"
    src_video_dir = args.video_dir if args.video_dir is not None else args.input_dir / "debug_video"
    aug_meta_path = args.input_dir / "augmentation_metadata.json"
    output_dir = args.output_dir

    if not src_data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {src_data_dir}")
    if not aug_meta_path.exists():
        raise FileNotFoundError(f"Augmentation metadata not found: {aug_meta_path}")

    with open(aug_meta_path, "r") as f:
        aug_meta = json.load(f)

    # Camera mapping: source video suffix -> LeRobot video key
    camera_map = {
        "head_camera": "observation.images.top_head",
        "left_camera": "observation.images.hand_left",
        "right_camera": "observation.images.hand_right",
        "third_view": "observation.images.front_camera",
    }

    # Prepare LeRobot directories
    data_dir = output_dir / "data" / "chunk-000"
    video_dir = output_dir / "videos" / "chunk-000"
    meta_dir = output_dir / "meta"

    for d in [data_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)
    for cam_key in camera_map.values():
        (video_dir / cam_key).mkdir(parents=True, exist_ok=True)

    # Discover episodes
    hdf5_files = sorted(src_data_dir.glob("episode*.hdf5"))
    if not hdf5_files:
        raise ValueError(f"No episode*.hdf5 files found in {src_data_dir}")

    # Extract episode indices from filenames (e.g., episode0.hdf5 -> 0)
    episodes = []
    for hdf5_path in hdf5_files:
        stem = hdf5_path.stem  # "episode0"
        ep_idx = int(stem.replace("episode", ""))
        episodes.append((hdf5_path, ep_idx))

    episodes.sort(key=lambda x: x[1])

    total_frames = 0
    episode_infos = []
    global_index = 0

    for hdf5_path, ep_idx in episodes:
        ep_key = str(ep_idx)
        print(f"Processing {hdf5_path.name} -> episode_{ep_idx:06d}")

        with h5py.File(hdf5_path, "r") as f:
            n_frames = f["joint_action/vector"].shape[0]

            # State and action: both use joint_action/vector (14D)
            action = f["joint_action/vector"][:]  # (N, 14)
            state = action.copy().astype(np.float32)
            action = action.astype(np.float32)

            # Timestamp
            timestamp = (np.arange(n_frames, dtype=np.float32) / args.fps).reshape(-1, 1)

            # Indices
            frame_index = np.arange(n_frames, dtype=np.int64).reshape(-1, 1)
            episode_index_arr = np.full((n_frames, 1), ep_idx, dtype=np.int64)
            index_arr = np.arange(global_index, global_index + n_frames, dtype=np.int64).reshape(-1, 1)
            task_index_arr = np.zeros((n_frames, 1), dtype=np.int64)

            # Compute stage_progress_gt
            if ep_key in aug_meta:
                ep_meta = aug_meta[ep_key]
                subtask_indices = ep_meta.get("subtask_completion_indices", [])
                segments = ep_meta.get("segments", [])
                stage_progress_gt = compute_stage_progress_gt(
                    n_frames=n_frames,
                    subtask_completion_indices=subtask_indices,
                    segments=segments,
                    k_stages=3,
                )
            else:
                print(f"  Warning: episode {ep_key} not found in augmentation_metadata.json, falling back to linear K=1")
                stage_progress_gt = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)

            stage_progress_gt = stage_progress_gt.reshape(-1, 1)

            # Build DataFrame
            df = pd.DataFrame({
                "observation.state": [s.tolist() for s in state],
                "action": [a.tolist() for a in action],
                "timestamp": timestamp.flatten().tolist(),
                "frame_index": frame_index.flatten().tolist(),
                "episode_index": episode_index_arr.flatten().tolist(),
                "index": index_arr.flatten().tolist(),
                "task_index": task_index_arr.flatten().tolist(),
                "stage_progress_gt": stage_progress_gt.flatten().tolist(),
            })

            # Write parquet
            parquet_path = data_dir / f"episode_{ep_idx:06d}.parquet"
            df.to_parquet(parquet_path, index=False)

            # Copy and rename videos
            for src_cam, dst_cam in camera_map.items():
                src_video = src_video_dir / f"episode{ep_idx}_{src_cam}.mp4"
                dst_video = video_dir / dst_cam / f"episode_{ep_idx:06d}.mp4"
                if src_video.exists():
                    shutil.copy2(src_video, dst_video)
                else:
                    print(f"  Warning: {src_video} not found!")

            episode_infos.append({
                "episode_index": ep_idx,
                "chunk_index": 0,
                "start_index": global_index,
                "end_index": global_index + n_frames - 1,
                "length": n_frames,
            })
            total_frames += n_frames
            global_index += n_frames

    # Build features schema
    video_feature_info = {
        "video.height": args.video_height,
        "video.width": args.video_width,
        "video.codec": args.video_codec,
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": args.fps,
        "video.channels": 3,
        "has_audio": False,
    }

    features = {
        "observation.images.top_head": {
            "dtype": "video",
            "shape": [args.video_height, args.video_width, 3],
            "names": ["height", "width", "channel"],
            "info": video_feature_info,
        },
        "observation.images.hand_left": {
            "dtype": "video",
            "shape": [args.video_height, args.video_width, 3],
            "names": ["height", "width", "channel"],
            "info": video_feature_info,
        },
        "observation.images.hand_right": {
            "dtype": "video",
            "shape": [args.video_height, args.video_width, 3],
            "names": ["height", "width", "channel"],
            "info": video_feature_info,
        },
        "observation.images.front_camera": {
            "dtype": "video",
            "shape": [args.video_height, args.video_width, 3],
            "names": ["height", "width", "channel"],
            "info": video_feature_info,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [14],
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": [14],
            "names": None,
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [1],
            "names": None,
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "task_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "stage_progress_gt": {
            "dtype": "float32",
            "shape": [1],
            "names": None,
        },
    }

    info = {
        "codebase_version": "v2.1",
        "robot_type": "agilex",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(episodes) * len(camera_map),
        "total_chunks": 1,
        "chunks_size": len(episodes),
        "fps": args.fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep_info in episode_infos:
            f.write(json.dumps(ep_info) + "\n")

    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": "manipulate blocks"}) + "\n")

    print(f"\nDone! Output at {output_dir}")
    print(f"Total episodes: {len(episodes)}, Total frames: {total_frames}")


if __name__ == "__main__":
    main()
