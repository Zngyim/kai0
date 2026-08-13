#!/usr/bin/env python3
"""Convert RoboTwin HDF5 demo data to LeRobot v2.1 format.

This script processes datasets which contain:
  - data/episode*.hdf5  (with joint_action/vector and camera rgb byte strings)
  - video/episode*.mp4  (optional, not used if HDF5 has rgb data)

Camera mapping:
  - head_camera  -> observation.images.top_head
  - left_camera  -> observation.images.hand_left
  - right_camera -> observation.images.hand_right

uv run python scripts/convert_robotwin_to_lerobot.py \
      --input-dir /mnt/pfs/zhangjiyao/yiming/RoboTwin/data/stack_blocks_three/demo_clean \
      --output-dir /mnt/pfs/zhangjiyao/yiming/kai0/data/stack_blocks_three/demo_clean_lerobot
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import io


def decode_rgb_bytes(rgb_bytes: np.ndarray) -> np.ndarray:
    """Decode a numpy array of JPEG bytes to RGB images."""
    images = []
    for b in rgb_bytes:
        img = Image.open(io.BytesIO(b))
        img = img.convert("RGB")
        images.append(np.array(img))
    return images


def write_video(frames: list, out_path: Path, fps: int = 30):
    """Write a list of RGB frames to an MP4 video using OpenCV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Convert RoboTwin HDF5 to LeRobot v2.1 format.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video-height", type=int, default=240)
    parser.add_argument("--video-width", type=int, default=320)
    parser.add_argument("--video-codec", type=str, default="h264")
    args = parser.parse_args()

    src_data_dir = args.input_dir / "data"
    output_dir = args.output_dir

    # Camera mapping from HDF5 groups to LeRobot video keys
    camera_map = {
        "head_camera": "observation.images.top_head",
        "left_camera": "observation.images.hand_left",
        "right_camera": "observation.images.hand_right",
    }

    data_dir = output_dir / "data" / "chunk-000"
    video_dir = output_dir / "videos" / "chunk-000"
    meta_dir = output_dir / "meta"

    for d in [data_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)
    for cam_key in camera_map.values():
        (video_dir / cam_key).mkdir(parents=True, exist_ok=True)

    hdf5_files = sorted(src_data_dir.glob("episode*.hdf5"))
    if not hdf5_files:
        raise ValueError(f"No episode*.hdf5 files found in {src_data_dir}")

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
        print(f"Processing {hdf5_path.name} -> episode_{ep_idx:06d}")

        with h5py.File(hdf5_path, "r") as f:
            n_frames = f["joint_action/vector"].shape[0]
            action = f["joint_action/vector"][:].astype(np.float32)
            state = action.copy()

            timestamp = (np.arange(n_frames, dtype=np.float32) / args.fps).reshape(-1, 1)
            frame_index = np.arange(n_frames, dtype=np.int64).reshape(-1, 1)
            episode_index_arr = np.full((n_frames, 1), ep_idx, dtype=np.int64)
            index_arr = np.arange(global_index, global_index + n_frames, dtype=np.int64).reshape(-1, 1)
            task_index_arr = np.zeros((n_frames, 1), dtype=np.int64)

            df = pd.DataFrame({
                "observation.state": [s.tolist() for s in state],
                "action": [a.tolist() for a in action],
                "timestamp": timestamp.flatten().tolist(),
                "frame_index": frame_index.flatten().tolist(),
                "episode_index": episode_index_arr.flatten().tolist(),
                "index": index_arr.flatten().tolist(),
                "task_index": task_index_arr.flatten().tolist(),
            })

            parquet_path = data_dir / f"episode_{ep_idx:06d}.parquet"
            df.to_parquet(parquet_path, index=False)

            # Generate videos from HDF5 rgb byte strings
            for src_cam, dst_cam in camera_map.items():
                rgb_dataset = f[f"observation/{src_cam}/rgb"]
                print(f"  Decoding {src_cam} ({len(rgb_dataset)} frames)...")
                frames = decode_rgb_bytes(rgb_dataset[:])
                dst_video = video_dir / dst_cam / f"episode_{ep_idx:06d}.mp4"
                write_video(frames, dst_video, fps=args.fps)
                print(f"  Written {dst_video}")

            episode_infos.append({
                "episode_index": ep_idx,
                "chunk_index": 0,
                "start_index": global_index,
                "end_index": global_index + n_frames - 1,
                "length": n_frames,
            })
            total_frames += n_frames
            global_index += n_frames

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
        f.write(json.dumps({"task_index": 0, "task": "stack blocks"}) + "\n")

    print(f"\nDone! Output at {output_dir}")
    print(f"Total episodes: {len(episodes)}, Total frames: {total_frames}")


if __name__ == "__main__":
    main()
