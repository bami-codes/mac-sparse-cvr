"""Guards of the study C final evaluation that act before any model is loaded (no LightGBM / PyTorch scoring)."""

import json
from pathlib import Path

import pytest

from cvr import final_eval_c as F
from cvr.checks import CheckFailed


def _frozen_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    for d in ("models", "vocab"):
        (run / d).mkdir(parents=True)
    (run / "models" / "G-shared_s2026.txt").write_text("model")
    (run / "vocab" / "ad_feat1.parquet").write_bytes(b"v")
    for f in ("selection.json", "results.json", "latency_c.json", "diagnostics_calib.json", "diagnostics_star.json",
              "frozen_domain_stats.parquet", "frozen_g1_stats.parquet", "config.toml"):
        (run / f).write_text("{}")
    manifest = {"config": {"data": {"mode": "files"}}, "selection": {}, "seeds": [2026, 2027, 2028],
                "holdout_files": F.HOLDOUT, "files_sha256": F.hashed_files(run)}
    (run / "freeze.json").write_text(json.dumps(manifest))
    return run


def test_freeze_is_never_repeated(tmp_path):
    run = _frozen_run(tmp_path)
    with pytest.raises(CheckFailed, match="never re-frozen"):
        F.freeze(run)


def test_any_changed_frozen_file_stops_the_evaluation(tmp_path):
    run = _frozen_run(tmp_path)
    F.verify_freeze(run)
    (run / "models" / "G-shared_s2026.txt").write_text("changed")
    with pytest.raises(CheckFailed, match="changed since freeze"):
        F.verify_freeze(run)


def test_test_mode_refuses_before_loading_anything(tmp_path, monkeypatch):
    run = _frozen_run(tmp_path)
    monkeypatch.setattr(F, "git_state", lambda: {"commit": "x", "dirty_files": []})
    holdout = [tmp_path / "train-13.parquet", tmp_path / "train-14.parquet"]
    with pytest.raises(CheckFailed, match="dry run must pass"):
        F.evaluate(run, holdout, "test")
    dry = run / "final" / "dryrun"
    dry.mkdir(parents=True)
    (dry / "results.json").write_text(json.dumps({"dryrun_passed": True}))
    with pytest.raises(CheckFailed, match="differ from the frozen holdout"):
        F.evaluate(run, [tmp_path / "train-12.parquet", tmp_path / "train-14.parquet"], "test")
    (run / "final" / "test").mkdir(parents=True)
    with pytest.raises(CheckFailed, match="evaluated only once"):
        F.evaluate(run, holdout, "test")
    monkeypatch.setattr(F, "git_state", lambda: {"commit": "x", "dirty_files": [" M cvr/x.py"]})
    (run / "final" / "test").rmdir()
    with pytest.raises(CheckFailed, match="uncommitted changes"):
        F.evaluate(run, holdout, "test")
