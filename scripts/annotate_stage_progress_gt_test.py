import json

import numpy as np
import pytest

from scripts import annotate_stage_progress_gt as annotate


def test_compute_linear_progress_gt():
    np.testing.assert_allclose(annotate.compute_linear_progress_gt(5), [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_array_equal(annotate.compute_linear_progress_gt(1), [0.0])
    with pytest.raises(ValueError, match="positive"):
        annotate.compute_linear_progress_gt(0)


def test_update_metadata(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "total_videos": 3,
                "features": {
                    "image": {"dtype": "video"},
                    "extra_view_image": {"dtype": "video"},
                    "extra_view_image-0": {"dtype": "video"},
                },
            }
        )
    )
    (meta_dir / "episodes_stats.jsonl").write_text(
        json.dumps({"episode_index": 0, "stats": {"extra_view_image-0": {"count": [3]}}}) + "\n"
    )
    (meta_dir / "stats.json").write_text(json.dumps({"extra_view_image-0": {"count": [3]}}))

    annotate.update_metadata(
        tmp_path,
        labels_by_episode={0: np.asarray([0.0, 0.5, 1.0], dtype=np.float32)},
        excluded_features=["extra_view_image-0"],
    )

    info = json.loads((meta_dir / "info.json").read_text())
    assert info["total_videos"] == 2
    assert "extra_view_image-0" not in info["features"]
    assert info["features"]["stage_progress_gt"]["dtype"] == "float32"

    episode_stats = json.loads((meta_dir / "episodes_stats.jsonl").read_text())
    assert "extra_view_image-0" not in episode_stats["stats"]
    assert episode_stats["stats"]["stage_progress_gt"]["min"] == [0.0]
    assert episode_stats["stats"]["stage_progress_gt"]["max"] == [1.0]

    stats = json.loads((meta_dir / "stats.json").read_text())
    assert "extra_view_image-0" not in stats
    assert stats["stage_progress_gt"]["count"] == [3]
