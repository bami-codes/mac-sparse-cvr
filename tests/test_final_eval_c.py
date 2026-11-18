"""Guards of the study C final evaluation that act before any model is loaded (no LightGBM / PyTorch scoring)."""

import hashlib
import json
from pathlib import Path

import pytest

from cvr import final_eval_c as F
from cvr.checks import CheckFailed

CLEAN = {"commit": "x", "dirty_files": []}


def _holdout(tmp_path: Path, content: dict[str, bytes]) -> list[Path]:
    out = []
    for name, data in content.items():
        p = tmp_path / "data" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        out.append(p)
    return out


def _frozen_run(tmp_path: Path, official: dict[str, bytes]) -> Path:
    run = tmp_path / "run"
    for d in ("models", "vocab"):
        (run / d).mkdir(parents=True)
    (run / "models" / "G-shared_s2026.txt").write_text("model")
    (run / "vocab" / "ad_feat1.parquet").write_bytes(b"v")
    for f in ("selection.json", "latency_c.json", "diagnostics_calib.json", "diagnostics_star.json",
              "frozen_domain_stats.parquet", "frozen_g1_stats.parquet", "config.toml"):
        (run / f).write_text("{}")
    (run / "results.json").write_text(json.dumps({"meta": {"data": {"v2": [{"file": "/x/train-12.parquet",
                                                                            "sha256": V2_SHA}]}}}))
    exp = {n: {"size": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in official.items()}
    (run / "holdout_expected.json").write_text(json.dumps({"files": exp}))
    manifest = {"config": {"data": {"mode": "files"}}, "selection": {}, "seeds": [2026, 2027, 2028],
                "holdout_files": F.HOLDOUT, "holdout_expected": exp, "files_sha256": F.hashed_files(run)}
    (run / "freeze.json").write_text(json.dumps(manifest))
    return run


V2_SHA = "ab" * 32


def _dry(run: Path, **over) -> None:
    rec = {"dryrun_passed": True, "freeze_sha256": F.sha256(run / "freeze.json"),
           "code_fingerprint": F.code_fingerprint(), "evaluation_git": CLEAN,
           "files_sha256": {"train-12.parquet": V2_SHA}} | over
    (run / "final" / "dryrun").mkdir(parents=True, exist_ok=True)
    (run / "final" / "dryrun" / "results.json").write_text(json.dumps(rec))


OFFICIAL = {"train-13.parquet": b"official day 13", "train-14.parquet": b"official day 14"}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "git_state", lambda: CLEAN)
    run = _frozen_run(tmp_path, OFFICIAL)
    return run, _holdout(tmp_path, OFFICIAL)


def test_freeze_is_never_repeated(setup):
    run, _ = setup
    with pytest.raises(CheckFailed, match="never re-frozen"):
        F.freeze(run)


def test_any_changed_frozen_file_stops_the_evaluation(setup):
    run, _ = setup
    F.verify_freeze(run)
    (run / "models" / "G-shared_s2026.txt").write_text("changed")
    with pytest.raises(CheckFailed, match="changed since freeze"):
        F.verify_freeze(run)


def test_the_dry_run_must_belong_to_this_freeze_and_this_code(setup):
    run, files = setup
    with pytest.raises(CheckFailed, match="dry run must pass"):
        F.evaluate(run, files, "test")
    for over, msg in (({"freeze_sha256": "0" * 64}, "different freeze"),  # review: passed before
                      ({"code_fingerprint": "0" * 64}, "different evaluation code"),
                      ({"evaluation_git": {"commit": "x", "dirty_files": [" M cvr/x.py"]}}, "uncommitted tree"),
                      ({"files_sha256": {"train-12.parquet": "cd" * 32}}, "V2 files"),  # review: passed before
                      ({"files_sha256": {}}, "V2 files")):
        _dry(run, **over)
        with pytest.raises(CheckFailed, match=msg):
            F.evaluate(run, files, "test")
    assert not (run / "final" / "test.lock").exists()


def test_input_problems_are_refused_before_the_single_run_is_claimed(setup, tmp_path):
    run, files = setup
    _dry(run)
    with pytest.raises(CheckFailed, match="differ from the frozen holdout"):
        F.evaluate(run, [files[0], tmp_path / "data" / "train-12.parquet"], "test")
    with pytest.raises(CheckFailed, match="does not exist"):  # review: a missing file used the single run
        F.evaluate(run, [files[0], tmp_path / "missing" / "train-14.parquet"], "test")
    renamed = _holdout(tmp_path / "dev", {"train-13.parquet": b"development data", "train-14.parquet": OFFICIAL["train-14.parquet"]})
    with pytest.raises(CheckFailed, match="bytes|SHA-256"):  # review: development data under a holdout name passed
        F.evaluate(run, renamed, "test")
    same_size = _holdout(tmp_path / "dev2", {"train-13.parquet": b"official day 1X", "train-14.parquet": OFFICIAL["train-14.parquet"]})
    with pytest.raises(CheckFailed, match="SHA-256"):
        F.evaluate(run, same_size, "test")
    assert not (run / "final" / "test.lock").exists()  # every refusal above can be retried


def test_only_one_process_claims_the_run(setup):
    run, _ = setup
    F.claim_test(run)
    with pytest.raises(CheckFailed, match="another process"):  # review: two concurrent calls both started
        F.claim_test(run)


def test_a_failure_after_the_claim_is_recorded_and_uses_the_run(setup, monkeypatch):
    run, files = setup
    _dry(run)

    def boom(*a, **k):
        raise RuntimeError("failed while reading the holdout")

    monkeypatch.setattr(F, "_evaluate", boom)
    with pytest.raises(RuntimeError):
        F.evaluate(run, files, "test")
    status = json.loads((run / "final" / "test_status.json").read_text())
    assert status["state"] == "failed_after_claim" and (run / "final" / "test.lock").exists()
    with pytest.raises(CheckFailed, match="only once"):
        F.evaluate(run, files, "test")


def _lat_family(fam, seeds=(2026, 2027, 2028), p99=(0.1, 0.2, 0.3), summary=None, requests=5000, version=2,
                others=(), other_cpu=250.0, num_diff=0.0, accepted=True):
    tol = 1e-9 if fam.startswith("G-") else 1e-6
    env = {"load_average": [5.0, 5.0, 5.0], "cpu_count": 10, "other_repo_python": list(others),
           "other_cpu_percent": other_cpu}
    return {"protocol": {"version": version, "requests": requests}, "environment_start": env,
            "environment_end": dict(env), "accepted": accepted,
            "seeds": {str(s): {"consistency": {"codes_mismatch": 0, "num_max_abs_diff": num_diff,
                                               "prediction_max_abs_diff": 0.0, "tolerances": {"pred": tol}},
                               "batch1_traffic": {"p99_ms": v, "requests": requests},
                               "batch1_stratified": {"p99_ms": v}} for s, v in zip(seeds, p99)},
            "p99_ms_by_seed": summary or {str(s): v for s, v in zip(seeds, p99)}}


def test_frozen_p99_comes_from_an_accepted_detailed_record():
    ok = {f: _lat_family(f) for f in F.LATENCY_FAMILIES}
    assert F.frozen_p99(ok)[0]["G-shared"] == {2026: 0.1, 2027: 0.2, 2028: 0.3}
    bad = [ok | {"G-shared": _lat_family("G-shared", seeds=(2026,), p99=(0.1,))},  # review: one seed was frozen
           ok | {"N-head": _lat_family("N-head", p99=(0.1, -0.2, 0.3))},  # review: a negative latency
           ok | {"N-int": _lat_family("N-int", p99=(0.1, float("nan"), 0.3))},
           ok | {"G-int": _lat_family("G-int", summary={"2026": 0.1, "2027": 0.2, "2028": 0.05})},
           ok | {"G-int": _lat_family("G-int", summary={"2026": 0.1, "2027": float("nan"), "2028": 0.3})},  # review
           ok | {"G-shared": _lat_family("G-shared", num_diff=1.0)},  # review: a numeric-feature error of 1.0
           ok | {"G-shared": _lat_family("G-shared", requests=1000, version=1)},  # review: the excluded measurement
           ok | {"G-shared": _lat_family("G-shared", others=["/x/.venv/bin/python -m pytest"], accepted=True)},
           ok | {"G-shared": _lat_family("G-shared", other_cpu=450.0)},
           ok | {"G-shared": _lat_family("G-shared", accepted=False)}]
    for lat in bad:
        with pytest.raises(CheckFailed):
            F.frozen_p99(lat)


def test_dry_run_secondary_names_and_leave_one_out_must_match():
    names = F.secondary_names({"N-star - N-shared": 1, "N-head - G-shared": 2}, "N-head", "G-shared")
    assert names == {"N-star - N-shared": "N-star - N-shared", "best NN - best GBDT": "N-head - G-shared"}
    rec = {"full": -1e-5, "min": -2e-5, "max": 1e-5, "sign_flips": 1, "by_removed_domain": {"13": -2e-5, "14": 1e-5}}
    assert F.same_loo({**rec, "by_removed_domain": {13: -2e-5, 14: 1e-5}}, rec) == 5
    with pytest.raises(CheckFailed, match="domains"):  # review: an emptied per-domain table passed
        F.same_loo({**rec, "by_removed_domain": {}}, rec)
    with pytest.raises(CheckFailed, match="domain 14"):
        F.same_loo({**rec, "by_removed_domain": {13: -2e-5, 14: 3e-5}}, rec)
