#!/usr/bin/env python3
"""Evaluate a causal OOD score from the advantage estimator's progress output.

For each frame t, this script first predicts the absolute progress V_t from
the episode's initial frame and the current frame. It then computes the
causal score

    S_t = sum_{h in offsets} (V_t - V_{t-h}) / h.

The default offsets are 11, 12, 13, 14, and 15 frames. Only frames at or before
t are used in S_t, so the calculation can be replayed online by caching the
initial frame and the preceding max(offsets) value predictions.

By default, a frame is marked OOD when S_t is less than ``--threshold``, which
indicates that progress has stalled or regressed. Use ``--comparison greater``
only for a reversed-direction control experiment.

Example:
    uv run python scripts/eval_causal_ood_score.py \
        --ckpt-dir /path/to/checkpoint \
        --dataset /path/to/lerobot_dataset \
        --config-name STACK_BLOCKS_ADVANTAGE \
        --prompt "Stack three blocks." \
        --threshold 0.08 \
        --comparison less \
        --history-offsets 11 12 13 14 15 \
        --output-dir ./eval_causal_ood
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))


CAMERA_KEYS = (
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
)


def compute_causal_ood_score(
    progress: np.ndarray,
    history_offsets: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the weighted causal score and its per-history contributions.

    Args:
        progress: Per-frame predicted progress values V_t.
        history_offsets: Positive history offsets included in the score.

    Returns:
        ``(score, terms)`` where each term is ``(V_t - V_{t-h}) / h`` for
        one configured history offset. Scores before the maximum offset, or
        scores containing missing values, are NaN.
    """
    history_offsets = tuple(history_offsets)
    if not history_offsets:
        raise ValueError("history_offsets must not be empty")
    if any(offset < 1 for offset in history_offsets):
        raise ValueError(f"history_offsets must all be positive, got {history_offsets}")
    if len(set(history_offsets)) != len(history_offsets):
        raise ValueError(f"history_offsets must be unique, got {history_offsets}")

    frame_count = len(progress)
    terms = np.full((frame_count, len(history_offsets)), np.nan, dtype=np.float32)
    for term_index, history_offset in enumerate(history_offsets):
        current = progress[history_offset:]
        previous = progress[:-history_offset]
        terms[history_offset:, term_index] = (current - previous) / history_offset

    valid = np.all(np.isfinite(terms), axis=1)
    score = np.full(frame_count, np.nan, dtype=np.float32)
    score[valid] = terms[valid].sum(axis=1)
    return score, terms


def classify_ood(score: np.ndarray, threshold: float, comparison: str) -> np.ndarray:
    """Classify valid scores according to the selected threshold direction."""
    valid = np.isfinite(score)
    if comparison == "greater":
        return valid & (score > threshold)
    if comparison == "less":
        return valid & (score < threshold)
    raise ValueError(f"Unsupported comparison: {comparison}")


def compute_binary_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    """Return frame-level binary metrics without adding an sklearn dependency."""
    valid = np.isfinite(labels)
    predictions = predictions[valid]
    labels = labels[valid].astype(bool)

    true_positive = int(np.sum(predictions & labels))
    false_positive = int(np.sum(predictions & ~labels))
    false_negative = int(np.sum(~predictions & labels))
    true_negative = int(np.sum(~predictions & ~labels))

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else np.nan
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision + recall) and precision + recall
        else np.nan
    )

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def get_episode_paths(dataset_dir: Path, info: dict, episode_index: int) -> tuple[Path, tuple[Path, Path, Path]]:
    """Build the parquet and three camera-video paths for one LeRobot episode."""
    chunk_index = episode_index // info["chunks_size"]
    format_kwargs = {
        "episode_chunk": chunk_index,
        "episode_index": episode_index,
    }
    parquet_path = dataset_dir / info["data_path"].format(**format_kwargs)
    video_paths = tuple(
        dataset_dir / info["video_path"].format(video_key=camera_key, **format_kwargs) for camera_key in CAMERA_KEYS
    )
    return parquet_path, video_paths


def plot_episode(
    output_path: Path,
    episode_index: int,
    ground_truth_progress: np.ndarray | None,
    predicted_progress: np.ndarray,
    score: np.ndarray,
    terms: np.ndarray,
    ood_mask: np.ndarray,
    threshold: float,
    comparison: str,
    history_offsets: Sequence[int],
) -> None:
    """Save a progress, component, and OOD-score visualization for one episode."""
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    frame_indices = np.arange(len(predicted_progress))

    if ground_truth_progress is not None:
        axes[0].plot(ground_truth_progress, color="tab:blue", label="stage_progress_gt")
    axes[0].plot(predicted_progress, color="tab:green", label="predicted absolute progress")
    axes[0].set_ylabel("Progress")
    axes[0].set_title(f"Episode {episode_index}: causal OOD score")
    axes[0].grid(visible=True)
    axes[0].legend(loc="best")

    for term_index, history_offset in enumerate(history_offsets):
        axes[1].plot(terms[:, term_index], label=f"(V(t) - V(t-{history_offset})) / {history_offset}")
    axes[1].set_ylabel("Score term")
    axes[1].grid(visible=True)
    axes[1].legend(loc="best", ncol=2)

    axes[2].plot(score, color="tab:purple", label="causal OOD score")
    axes[2].axhline(threshold, color="tab:red", linestyle="--", label=f"threshold = {threshold:.4f}")
    axes[2].scatter(
        frame_indices[ood_mask],
        score[ood_mask],
        color="tab:red",
        marker="x",
        label=f"OOD: score {comparison} threshold",
        zorder=3,
    )
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("OOD score")
    axes[2].grid(visible=True)
    axes[2].legend(loc="best")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def evaluate_dataset(
    checkpoint_dir: Path,
    dataset_dir: Path,
    config_name: str,
    prompt: str,
    output_dir: Path,
    threshold: float,
    comparison: str,
    history_offsets: Sequence[int],
    batch_size: int,
    ood_label_column: str | None,
) -> None:
    """Run causal-score evaluation, save annotated parquets, plots, and a CSV summary."""
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {info_path}")

    with info_path.open(encoding="utf-8") as file_handle:
        info = json.load(file_handle)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Loading checkpoint: {checkpoint_dir}")
    from stage_advantage.annotation.evaluator import SimpleValueEvaluator

    evaluator = SimpleValueEvaluator(config_name=config_name, ckpt_dir=str(checkpoint_dir), num_workers=4)
    summaries: list[dict[str, float | int]] = []

    try:
        total_episodes = int(info["total_episodes"])
        print(f"[INFO] Evaluating {total_episodes} episodes from: {dataset_dir}")
        for episode_index in range(total_episodes):
            parquet_path, video_paths = get_episode_paths(dataset_dir, info, episode_index)
            if not parquet_path.exists() or not all(path.exists() for path in video_paths):
                print(f"[WARN] Episode {episode_index}: parquet or video missing, skipping")
                continue

            dataframe = pd.read_parquet(parquet_path)
            frame_count = len(dataframe)
            max_history_offset = max(history_offsets)
            if frame_count <= max_history_offset:
                print(f"[WARN] Episode {episode_index}: only {frame_count} frames, skipping")
                continue

            print(f"[INFO] Episode {episode_index}: inferring {frame_count} frames")
            results = evaluator.evaluate_video_2timesteps_advantages(
                video_paths=tuple(str(path) for path in video_paths),
                prompt=prompt,
                batch_size=batch_size,
                frame_interval=1,
                relative_interval=1,
                min_frame_index=0,
                max_frame_index=frame_count - 1,
            )

            predicted_progress = np.full(frame_count, np.nan, dtype=np.float32)
            for result in results:
                frame_index = result["frame_idx"]
                if 0 <= frame_index < frame_count:
                    predicted_progress[frame_index] = result["absolute_value"]

            score, terms = compute_causal_ood_score(predicted_progress, history_offsets)
            ood_mask = classify_ood(score, threshold, comparison)

            output_dataframe = dataframe.copy()
            output_dataframe["predicted_absolute_progress"] = predicted_progress
            for term_index, history_offset in enumerate(history_offsets):
                output_dataframe[f"causal_progress_term_{history_offset}"] = terms[:, term_index]
            output_dataframe["causal_ood_score"] = score
            output_dataframe["is_causal_ood"] = ood_mask

            chunk_index = episode_index // info["chunks_size"]
            parquet_output_dir = output_dir / "data" / f"chunk-{chunk_index:03d}"
            parquet_output_dir.mkdir(parents=True, exist_ok=True)
            output_dataframe.to_parquet(parquet_output_dir / parquet_path.name, index=False)

            ground_truth_progress = None
            score_correlation = np.nan
            if "stage_progress_gt" in dataframe:
                ground_truth_progress = dataframe["stage_progress_gt"].to_numpy(dtype=np.float32)
                ground_truth_score, _ = compute_causal_ood_score(ground_truth_progress, history_offsets)
                valid_score = np.isfinite(score) & np.isfinite(ground_truth_score)
                if valid_score.sum() > 1:
                    score_correlation = float(np.corrcoef(score[valid_score], ground_truth_score[valid_score])[0, 1])

            summary: dict[str, float | int] = {
                "episode_index": episode_index,
                "frame_count": frame_count,
                "valid_score_frames": int(np.isfinite(score).sum()),
                "ood_frames": int(ood_mask.sum()),
                "ood_ratio": float(ood_mask.mean()),
                "score_min": float(np.nanmin(score)),
                "score_max": float(np.nanmax(score)),
                "score_mean": float(np.nanmean(score)),
                "score_gt_correlation": score_correlation,
            }

            if ood_label_column is not None:
                if ood_label_column not in dataframe:
                    raise KeyError(f"OOD label column not found: {ood_label_column}")
                summary.update(compute_binary_metrics(ood_mask, dataframe[ood_label_column].to_numpy()))

            summaries.append(summary)
            plot_episode(
                output_path=output_dir / f"episode_{episode_index:06d}_causal_ood.png",
                episode_index=episode_index,
                ground_truth_progress=ground_truth_progress,
                predicted_progress=predicted_progress,
                score=score,
                terms=terms,
                ood_mask=ood_mask,
                threshold=threshold,
                comparison=comparison,
                history_offsets=history_offsets,
            )
            print(
                f"  [OK] OOD frames={summary['ood_frames']}/{frame_count}, "
                f"score=[{summary['score_min']:.4f}, {summary['score_max']:.4f}]"
            )
    finally:
        evaluator.shutdown()

    summary_path = output_dir / "causal_ood_summary.csv"
    summary_dataframe = pd.DataFrame(summaries)
    summary_dataframe.to_csv(summary_path, index=False)
    print(f"[OK] Summary written to: {summary_path}")


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True, help="Advantage estimator checkpoint directory.")
    parser.add_argument("--dataset", type=Path, required=True, help="LeRobot dataset directory.")
    parser.add_argument("--config-name", required=True, help="Advantage estimator training config name.")
    parser.add_argument("--prompt", required=True, help="Task prompt used by the advantage estimator.")
    parser.add_argument("--output-dir", type=Path, default=Path("./eval_causal_ood"))
    parser.add_argument("--threshold", type=float, default=0.08, help="OOD decision threshold (default: 0.08).")
    parser.add_argument(
        "--comparison",
        choices=("greater", "less"),
        default="less",
        help="Mark OOD if score is less (default) or greater than threshold.",
    )
    parser.add_argument(
        "--history-offsets",
        type=int,
        nargs="+",
        default=[11, 12, 13, 14, 15],
        help="History offsets used by the score (default: 11 12 13 14 15).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--ood-label-column",
        default=None,
        help="Optional parquet column containing binary OOD labels for precision/recall/F1.",
    )
    return parser.parse_args(arguments)


def main() -> None:
    """Run the command-line entry point."""
    args = parse_args()
    evaluate_dataset(
        checkpoint_dir=args.ckpt_dir,
        dataset_dir=args.dataset,
        config_name=args.config_name,
        prompt=args.prompt,
        output_dir=args.output_dir,
        threshold=args.threshold,
        comparison=args.comparison,
        history_offsets=args.history_offsets,
        batch_size=args.batch_size,
        ood_label_column=args.ood_label_column,
    )


if __name__ == "__main__":
    main()
