"""Guards of the study C final evaluation that act before any model is loaded (no LightGBM / PyTorch scoring)."""

import hashlib
import json
from pathlib import Path

import pytest

from cvr import final_eval_c as F
from cvr.checks import CheckFailed

CLEAN = {"commit": "c" * 40, "dirty_files": []}


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
    for over, msg in (({"freeze_sha256": "0" * 64}, "different freeze"),
                      ({"code_fingerprint": "0" * 64}, "different evaluation code"),
                      ({"evaluation_git": {"commit": "x", "dirty_files": [" M cvr/x.py"]}}, "uncommitted tree"),
                      ({"files_sha256": {"train-12.parquet": "cd" * 32}}, "V2 files"),
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
    with pytest.raises(CheckFailed, match="does not exist"):  # must not use up the single run
        F.evaluate(run, [files[0], tmp_path / "missing" / "train-14.parquet"], "test")
    renamed = _holdout(tmp_path / "dev", {"train-13.parquet": b"development data", "train-14.parquet": OFFICIAL["train-14.parquet"]})
    with pytest.raises(CheckFailed, match=r"bytes|SHA-256"):  # development data under a holdout name
        F.evaluate(run, renamed, "test")
    same_size = _holdout(tmp_path / "dev2", {"train-13.parquet": b"official day 1X", "train-14.parquet": OFFICIAL["train-14.parquet"]})
    with pytest.raises(CheckFailed, match="SHA-256"):
        F.evaluate(run, same_size, "test")
    assert not (run / "final" / "test.lock").exists()  # every refusal above can be retried


def test_outside_a_git_repository_the_test_is_refused(setup, monkeypatch):
    run, files = setup
    _dry(run)
    monkeypatch.setattr(F, "git_state", lambda: {"commit": "", "dirty_files": []})  # what git_state returns there
    with pytest.raises(CheckFailed, match="not inside a git repository"):
        F.evaluate(run, files, "test")
    monkeypatch.setattr(F, "git_state", lambda: CLEAN)
    _dry(run, evaluation_git={"commit": "", "dirty_files": []})
    with pytest.raises(CheckFailed, match="dry run is not inside a git repository"):
        F.evaluate(run, files, "test")
    assert not (run / "final" / "test.lock").exists()


def test_only_one_process_claims_the_run(setup):
    run, _ = setup
    F.claim_test(run)
    with pytest.raises(CheckFailed, match="another process"):  # two concurrent calls must not both start
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
    bad = [ok | {"G-shared": _lat_family("G-shared", seeds=(2026,), p99=(0.1,))},  # a single seed
           ok | {"N-head": _lat_family("N-head", p99=(0.1, -0.2, 0.3))},  # a negative latency
           ok | {"N-int": _lat_family("N-int", p99=(0.1, float("nan"), 0.3))},
           ok | {"G-int": _lat_family("G-int", summary={"2026": 0.1, "2027": 0.2, "2028": 0.05})},
           ok | {"G-int": _lat_family("G-int", summary={"2026": 0.1, "2027": float("nan"), "2028": 0.3})},  # a NaN summary
           ok | {"G-shared": _lat_family("G-shared", num_diff=1.0)},  # a numeric-feature error of 1.0
           ok | {"G-shared": _lat_family("G-shared", requests=1000, version=1)},  # the excluded earlier measurement
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
    with pytest.raises(CheckFailed, match="domains"):  # an emptied per-domain table
        F.same_loo({**rec, "by_removed_domain": {}}, rec)
    with pytest.raises(CheckFailed, match="domain 14"):
        F.same_loo({**rec, "by_removed_domain": {13: -2e-5, 14: 3e-5}}, rec)


# ---------------------------------------------------------------- latency provenance (§16 revision 8)
def _provenance_run(tmp_path):
    run = tmp_path / "prov"
    (run / "models").mkdir(parents=True)
    for fam in ("N-int", "N-shared", "N-head", "N-star"):
        for s in (2026, 2027, 2028):
            (run / "models" / f"{fam}_s{s}_e2.json").write_text("{}")
    sel = {"lambda_iiid": {f: 1.0 for f in F.LATENCY_FAMILIES}}
    from cvr.serving_c import model_path
    env = {"load_average": [5.0] * 3, "cpu_count": 10, "other_repo_python": [], "other_cpu_percent": 250.0}
    fams = {}
    for fam in F.LATENCY_FAMILIES:
        seeds = {str(s): {"server": {"model": str(model_path(run, fam, s)), "calibrator": f"calib_{fam}_s{s}",
                                     "lambda": 1.0, "group_key": "context_feat1", "calibration": "iiid"}}
                 for s in (2026, 2027, 2028)}
        p99 = {"2026": 0.1, "2027": 0.2, "2028": 0.3}
        fams[fam] = {"seeds": seeds, "p99_ms_by_seed": p99, "environment_start": env,
                     "attempts": [{"accepted": True, "environment_start": env, "environment_end": env, "p99_ms_by_seed": p99}]}
    return run, sel, {"run": str(run), "families": fams}


def test_latency_report_must_come_from_this_run_and_its_calibrators(tmp_path):
    import copy
    run, sel, rep = _provenance_run(tmp_path)
    F.check_latency_provenance(rep, run, sel)
    other = copy.deepcopy(rep)
    other["run"] = str(tmp_path / "smoke")  # a smoke run's report under the development run
    with pytest.raises(CheckFailed, match="latency report is for"):
        F.check_latency_provenance(other, run, sel)
    lam = copy.deepcopy(rep)
    lam["families"]["G-shared"]["seeds"]["2026"]["server"]["lambda"] = 100.0  # the served lambda differs from the chosen one (100 vs 1)
    with pytest.raises(CheckFailed, match="measured on"):
        F.check_latency_provenance(lam, run, sel)
    cal = copy.deepcopy(rep)
    cal["families"]["G-shared"]["seeds"]["2027"]["server"]["calibrator"] = "calib_G-int_s2027"
    with pytest.raises(CheckFailed, match="measured on"):
        F.check_latency_provenance(cal, run, sel)


def test_the_kept_latency_attempt_is_the_first_accepted_one(tmp_path):
    import copy
    run, sel, rep = _provenance_run(tmp_path)
    busy = {"load_average": [5.0] * 3, "cpu_count": 10, "other_repo_python": [], "other_cpu_percent": 460.0}
    ok_env = rep["families"]["N-star"]["environment_start"]
    good = copy.deepcopy(rep)
    good["families"]["N-star"]["attempts"] = [
        {"accepted": False, "environment_start": ok_env, "environment_end": busy, "p99_ms_by_seed": {"2026": 0.5}},
        rep["families"]["N-star"]["attempts"][0]]
    F.check_latency_provenance(good, run, sel)  # rejected, then accepted: the second one is kept
    two_ok = copy.deepcopy(rep)
    two_ok["families"]["N-star"]["attempts"] = [rep["families"]["N-star"]["attempts"][0]] * 2
    none_ok = copy.deepcopy(rep)
    none_ok["families"]["N-star"]["attempts"][0]["environment_end"] = busy
    other_p99 = copy.deepcopy(rep)
    other_p99["families"]["N-star"]["p99_ms_by_seed"] = {"2026": 0.1, "2027": 0.2, "2028": 0.05}
    for bad in (two_ok, none_ok, other_p99):
        with pytest.raises(CheckFailed, match="attempt"):
            F.check_latency_provenance(bad, run, sel)


def test_at_most_three_latency_attempts_and_the_same_rule_as_the_measuring_script(tmp_path):
    import copy

    import scripts.latency_c as L
    assert (
        L.MAX_ATTEMPTS, L.REQUESTS, L.ACCEPT_OTHER_CPU, L.PROTOCOL["version"]) == (F.LATENCY_MAX_ATTEMPTS, F.LATENCY_REQUESTS, F.ACCEPT_OTHER_CPU, F.LATENCY_PROTOCOL_VERSION)
    run, sel, rep = _provenance_run(tmp_path)
    busy = {"load_average": [5.0] * 3, "cpu_count": 10, "other_repo_python": [], "other_cpu_percent": 460.0}
    ok = rep["families"]["G-int"]["attempts"][0]
    rejected = {"accepted": False, "environment_start": busy, "environment_end": busy, "p99_ms_by_seed": {"2026": 0.5}}
    third = copy.deepcopy(rep)
    third["families"]["G-int"]["attempts"] = [rejected, rejected, ok]
    F.check_latency_provenance(third, run, sel)  # accepted on the last allowed attempt
    fourth = copy.deepcopy(rep)
    fourth["families"]["G-int"]["attempts"] = [rejected, rejected, rejected, ok]  # measured again until accepted
    with pytest.raises(CheckFailed, match="at most 3"):
        F.check_latency_provenance(fourth, run, sel)


def test_a_failed_process_table_is_a_failed_measurement(monkeypatch):
    import subprocess

    import scripts.latency_c as L
    monkeypatch.setattr(L.sys, "platform", "darwin")  # the rule is macOS-only; off macOS it is refused before ps runs
    monkeypatch.setattr(L.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="x"))
    with pytest.raises(RuntimeError, match="process table unavailable"):  # an empty table must not read as 0% CPU
        L.environment()
