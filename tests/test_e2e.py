"""End to end on synthetic MAC-shaped files: every command of both studies, as it is run on the real data.

    uv run pytest -m e2e          # a few minutes; not part of the default test run

The real data are not needed (tests/synthetic_mac.py writes train-08..14 with SHA-256 sidecars). Each command runs
as a subprocess from a temporary working directory that is its own git repository (data/, runs/ and configs/ are
ignored there), because the evaluation code checks a clean tree in the working directory; PYTHONPATH points at this
repository, so the code under test is this checkout's.

Study C: development run -> diagnostics (independent re-selection, STAR statistics) -> latency with the serving
check -> expected holdout sizes and hashes -> freeze -> V2 dry run -> refusals that must not use the single run
(no git repository, uncommitted file, wrong holdout file) -> the single test -> refusal of a second test and of a changed frozen file.
First study: pipeline (GBDT, logistic regression, M3 networks) -> freeze -> dry run -> test outside git refused ->
test -> second test refused.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from synthetic_mac import make_mac_tree

pytestmark = pytest.mark.e2e
REPO = Path(__file__).resolve().parents[1]
DATA = Path("data/raw/mac/train")
NO_GIT = {"GIT_DIR": "/nonexistent"}  # git fails as outside a repository: the test must not treat that as clean

LGB = """
[models.lgb]
learning_rate = 0.05
num_leaves = 31
min_data_in_leaf = 100
feature_fraction = 0.9
bagging_fraction = 0.9
bagging_freq = 1
num_threads = 4
"""
STUDY_C = f"""
name = "c_e2e"
[data]
root = "{DATA}"
mode = "files"
train = ["train-08.parquet", "train-09.parquet", "train-10.parquet"]
v1 = ["train-11.parquet"]
v2 = ["train-12.parquet"]
[vocab]
min_count = 1
[domains]
small_share = 0.01
small_min_clicks = 150
[groups]
n_min = 400
c_min = 40
c_dense = 100
[models]
seeds = [2026, 2027, 2028]
num_boost_round = 300
early_stopping_rounds = 20
{LGB}cat_smooth = 50.0
min_data_per_group = 200
[models.nn]
lrs = [0.001, 0.003]
epochs = [1, 2]
dim = 8
batch = 1024
threads = 4
[calibration]
lambdas = [1.0, 10.0, 100.0, 1000.0, 10000.0]
[bootstrap]
reps = 100
seed = 20261118
"""
FIRST = f"""
name = "first_e2e"
[data]
root = "{DATA}"
mode = "files"
train = ["train-08.parquet", "train-09.parquet", "train-10.parquet"]
v1 = ["train-11.parquet"]
v2 = ["train-12.parquet"]
[vocab]
min_count = 1
[groups]
n_min = 400
c_min = 40
c_dense = 100
[models]
seeds = [2026, 2027, 2028]
lr_C = 1.0
lr_max_iter = 2000
num_boost_round = 300
early_stopping_rounds = 20
ablation_no_history = true
{LGB}
[[models.lgb_candidates]]
name = "default"
[[models.lgb_candidates]]
name = "smooth50"
cat_smooth = 50.0
min_data_per_group = 200
[models.m3]
enabled = true
variants = ["attn", "mean"]
dim = 8
epochs = 1
batch = 1024
lr = 0.003
id_min_count = 2
device = "cpu"
threads = 4
[calibration]
lambdas = [1.0, 10.0, 100.0, 1000.0, 10000.0]
[bootstrap]
reps = 100
seed = 20261117
"""


def git(ws: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=e2e", "-c", "user.email=e2e@example.invalid", "-c", "commit.gpgsign=false", *args],
        cwd=ws,
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("e2e")
    make_mac_tree(root / DATA)
    (root / "configs").mkdir()
    (root / "configs" / "c_e2e.toml").write_text(STUDY_C)
    (root / "configs" / "first_e2e.toml").write_text(FIRST)
    (root / ".gitignore").write_text("data/\nruns/\nconfigs/\nfake/\n")
    git(root, "init", "-q")
    git(root, "add", ".gitignore")
    git(root, "commit", "-q", "-m", "e2e workspace")
    return root


def run(ws: Path, *args: str, ok: bool = True, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = os.environ | {"PYTHONPATH": str(REPO), "PYTHONHASHSEED": "0"} | (env_extra or {})
    p = subprocess.run([sys.executable, *args], cwd=ws, env=env, capture_output=True, text=True, timeout=3600)
    if ok and p.returncode != 0:
        pytest.fail(
            f"{' '.join(args)} exited {p.returncode}\n--- stdout\n{p.stdout[-3000:]}\n--- stderr\n{p.stderr[-6000:]}"
        )
    if not ok and p.returncode == 0:
        pytest.fail(f"{' '.join(args)} was expected to be refused\n{p.stdout[-3000:]}")
    return p


def only_run(ws: Path, name: str) -> Path:
    runs = sorted((ws / "runs" / name).iterdir())
    assert len(runs) == 1, runs
    return runs[0].relative_to(ws)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_study_c_end_to_end(ws: Path) -> None:
    run(ws, "-m", "experiments.study_c", "configs/c_e2e.toml")
    r = only_run(ws, "c_e2e")
    res = json.loads((ws / r / "results.json").read_text())
    sel = res["selection"]
    assert res["meta"]["rows"] == {"train": 60_000, "v1": 20_000, "v2": 20_000}
    assert all(x["sha256"] for role in res["meta"]["data"].values() for x in role)  # sidecars were read
    assert (res["domains"]["small"]["domains"], res["domains"]["tiny"]["domains"]) == (6, 2)
    assert sel["candidate"] in ("G-shared", "N-int", "N-shared", "N-head", "N-star") and sel["control"] in (
        "G-int",
        "N-int",
    )

    run(ws, str(REPO / "experiments" / "scripts" / "study_c_diagnostics.py"), "calib", str(r))
    run(ws, str(REPO / "experiments" / "scripts" / "study_c_diagnostics.py"), "star", str(r))
    diag = json.loads((ws / r / "diagnostics_calib.json").read_text())["calib"]
    assert diag["candidate_recomputed_exact"] == sel["candidate"]

    if sys.platform != "darwin":  # the acceptance rule uses macOS `ps` (scripts/latency_c.environment)
        pytest.skip("the domain experiment's latency rule is defined on macOS; freeze needs an accepted measurement")
    run(ws, str(REPO / "experiments" / "scripts" / "latency_c.py"), str(r))
    lat = json.loads((ws / r / "latency_c.json").read_text())["families"]
    for f in lat.values():  # the serving check runs before any timing and is a hard check
        assert all(s["consistency"]["codes_mismatch"] == 0 for s in f["seeds"].values())
    if not all(f["accepted"] for f in lat.values()):
        p = run(ws, "-m", "experiments.final_eval_c", "freeze", str(r), ok=False)
        assert "acceptance rule" in p.stderr or "attempts" in p.stderr
        pytest.skip("this machine was too busy for an accepted latency measurement; freeze refused it as it should")

    files = {n: ws / DATA / n for n in ("train-13.parquet", "train-14.parquet")}
    (ws / r / "holdout_expected.json").write_text(
        json.dumps(
            {
                "source": "synthetic (tests/test_e2e.py)",
                "files": {n: {"size": f.stat().st_size, "sha256": sha256(f)} for n, f in files.items()},
            }
        )
    )
    run(ws, "-m", "experiments.final_eval_c", "freeze", str(r))
    assert "never re-frozen" in run(ws, "-m", "experiments.final_eval_c", "freeze", str(r), ok=False).stderr
    run(ws, "-m", "experiments.final_eval_c", "dryrun", str(r))
    dry = json.loads((ws / r / "final" / "dryrun" / "results.json").read_text())
    assert dry["dryrun_passed"] and dry["dryrun_values_reproduced"] > 0

    test = ["-m", "experiments.final_eval_c", "test", str(r)]
    holdout = [str(DATA / n) for n in files]
    lock = ws / r / "final" / "test.lock"
    assert "not inside a git repository" in run(ws, *test, *holdout, ok=False, env_extra=NO_GIT).stderr
    (ws / "notes.txt").write_text("uncommitted")  # a refusal before the claim leaves the single run unused
    assert "uncommitted changes" in run(ws, *test, *holdout, ok=False).stderr
    (ws / "notes.txt").unlink()
    (ws / "fake").mkdir()
    shutil.copy(ws / DATA / "train-12.parquet", ws / "fake" / "train-13.parquet")  # right name, wrong file
    p = run(ws, *test, "fake/train-13.parquet", holdout[1], ok=False)
    assert "official" in p.stderr
    assert (
        "differ from the frozen holdout" in run(ws, *test, str(DATA / "train-12.parquet"), holdout[1], ok=False).stderr
    )
    assert not lock.exists()

    run(ws, *test, *holdout)
    out = json.loads((ws / r / "final" / "test" / "results.json").read_text())
    assert json.loads((ws / r / "final" / "test_status.json").read_text())["state"] == "done"
    assert out["rows"] == 40_000 and out["files_sha256"] == {n: sha256(f) for n, f in files.items()}
    assert out["verdict"]["tier"] in ("adoption_candidate", "improvement_confirmed", "not_confirmed")
    assert {t["family"] for t in out["table"]} >= {"G-int", "G-shared", "N-star", "N-star-noPN"}
    assert "only once" in run(ws, *test, *holdout, ok=False).stderr

    calib = next((ws / r / "models").glob("calib_G-int_s2026*"))
    calib.write_bytes(calib.read_bytes() + b" ")
    assert "frozen files changed" in run(ws, "-m", "experiments.final_eval_c", "dryrun", str(r), ok=False).stderr


def test_first_study_end_to_end(ws: Path) -> None:
    run(ws, "-m", "experiments.pipeline", "configs/first_e2e.toml")
    r = only_run(ws, "first_e2e")
    res = json.loads((ws / r / "results.json").read_text())
    assert res["meta"]["rows"]["train"] == 60_000
    assert any(k.startswith("M3_attn_s2026|") for k in res["v2_rows"])
    run(ws, "-m", "experiments.final_eval", "freeze", str(r))
    frozen = json.loads((ws / r / "freeze.json").read_text())
    assert frozen["groups"]["G1_groups"] > 0
    run(ws, "-m", "experiments.final_eval", "dryrun", str(r))
    dry = json.loads((ws / r / "final" / "dryrun" / "results.json").read_text())
    assert dry["dryrun_metrics_reproduced"] > 0
    first_test = ["-m", "experiments.final_eval", "test", str(r), str(DATA / "train-13.parquet")]
    assert "not inside a git repository" in run(ws, *first_test, ok=False, env_extra=NO_GIT).stderr
    run(ws, *first_test)
    out = json.loads((ws / r / "final" / "test" / "results.json").read_text())
    assert out["rows"] == 20_000 and out["eval_file_sha256"] == sha256(ws / DATA / "train-13.parquet")
    assert isinstance(out["primary"]["success"], bool) and "m3_attn_vs_mean" in out["secondary"]
    assert "only once" in run(ws, *first_test, ok=False).stderr
