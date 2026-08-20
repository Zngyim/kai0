import scripts.train_pytorch as train_pytorch


def test_prune_old_checkpoints_keeps_latest_and_periodic(tmp_path):
    for name in ["5000", "10000", "15000", "20000", "25000", "tmp_30000", "notes"]:
        (tmp_path / name).mkdir()

    train_pytorch.prune_old_checkpoints(tmp_path, latest_step=25000, keep_period=10000)

    assert {path.name for path in tmp_path.iterdir()} == {"10000", "20000", "25000", "tmp_30000", "notes"}


def test_prune_old_checkpoints_without_period_keeps_only_latest(tmp_path):
    for step in [1, 2, 3]:
        (tmp_path / str(step)).mkdir()

    train_pytorch.prune_old_checkpoints(tmp_path, latest_step=3, keep_period=None)

    assert {path.name for path in tmp_path.iterdir()} == {"3"}
