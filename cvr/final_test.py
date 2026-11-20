"""Choose the settings, freeze, dry-run on V2, then evaluate the official test once per question.

    uv run python -m cvr.final_test choose runs/train/summary_grid.jsonl     # by train-19 GAUC
    uv run python -m cvr.evaluation calibrate runs/eval/config_choice.json  # Q2 calibrations on V1, lambdas on V2
    uv run python -m cvr.final_test freeze                                      # freeze_q1.json, freeze_q2.json
    uv run python -m cvr.final_test dryrun                                      # V2 values reproduced (hard check)
    (pod) python -m cvr.pymal_run predict --official_test freeze_q1.json --ckpt ... test.parquet   # one claim each
    uv run python -m cvr.final_test test q1
    uv run python -m cvr.final_test test q2

The freeze manifests hash the checkpoints, calibrators, group lists and selection. The official test is scored only
with a manifest that lists every checkpoint and the test file's SHA-256, and each question is evaluated once under an
exclusive lock. Before that, a dry run on V2 must pass for the same freeze and the same evaluation code, committed on a
clean tree. The rules are in docs/DESIGN.md §18.5-18.6.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import evaluation, groups, metrics
from .checks import CheckFailed, require_close, require_commit

PAPER = {"BASE": {"auc": 0.8494, "gauc": 0.7424}, "MoAE": {"auc": 0.8694, "gauc": 0.7636}}  # Table 4, last-click
TOL = {"reproduced": 0.002, "close": 0.005}
TIE = 1e-6
DRYRUN_TOL = 1e-9
FROZEN = groups.OUT / "frozen"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty_files": run("status", "--porcelain").splitlines()}


def code_fingerprint() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


# ------------------------------------------------------------------ settings chosen on train-19
def planned_grid() -> list[dict]:
    """The 45 planned grid jobs, taken from the queue's own definition (scripts/pod_queue.py grid)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "pod_queue", Path(__file__).resolve().parents[1] / "scripts" / "pod_queue.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.grid()


def check_complete(rows: list[dict]) -> dict:
    """Refuse to choose from a partial grid. The queue logs a failed job and moves on, so a missing job is easy to miss.

    Every planned job must appear once with its planned settings and either a finite validation GAUC/AUC or a
    recorded non-finite loss. Anything else, such as an infrastructure failure, has to be resumed first."""
    plan = {j["id"]: j for j in planned_grid()}
    seen: dict[str, int] = {}
    for r in rows:
        seen[r["id"]] = seen.get(r["id"], 0) + 1
    problems = [f"missing {i}" for i in plan if i not in seen]
    problems += [f"unplanned {i}" for i in seen if i not in plan]
    problems += [f"{i} appears {n} times" for i, n in seen.items() if n > 1]
    for r in rows:
        j = plan.get(r["id"])
        if j is None:
            continue
        if any(r[k] != j[k] for k in ("model", "lr", "lamda", "seed")):
            problems.append(f"{r['id']}: settings differ from the plan")
        if r["state"] == "done":
            if not all(
                np.isfinite(r.get(k, np.nan) if r.get(k) is not None else np.nan) for k in ("val_gauc", "val_auc")
            ):
                problems.append(f"{r['id']}: done without a finite validation GAUC/AUC")
        elif r["state"] == "diverged":
            if str((r.get("diverged") or {}).get("loss")) not in ("nan", "inf", "-inf"):
                problems.append(f"{r['id']}: stopped without a non-finite loss on record")
        else:
            problems.append(f"{r['id']}: state {r['state']} (resume it)")
    if problems:
        raise CheckFailed(f"the grid is not complete ({len(problems)} problems): {problems[:12]}")
    return {
        "jobs": len(plan),
        "validated": sum(r["state"] == "done" for r in rows),
        "diverged": sum(r["state"] == "diverged" for r in rows),
    }


def choose(summary: Path) -> dict:
    rows = [json.loads(line) for line in Path(summary).read_text().splitlines() if line.strip()]
    completeness = check_complete(rows)
    out = {"completeness": completeness}
    for model in ("MoAE", "BASE"):
        cand = [
            r
            for r in rows
            if r["model"] == model
            and r["seed"] == 2026
            and r["state"] == "done"
            and r.get("val_gauc") is not None
            and np.isfinite(r["val_gauc"])
        ]
        if not cand:
            raise CheckFailed(f"no validated {model} run")
        top = max(r["val_gauc"] for r in cand)
        tied = [r for r in cand if top - r["val_gauc"] < TIE]
        best_auc = max(r["val_auc"] for r in tied)
        tied = [r for r in tied if best_auc - r["val_auc"] < TIE]
        best = min(tied, key=lambda r: (abs(r["lr"] - 0.003), abs(r["lamda"] - 0.3)))
        out[model] = {
            "lr": best["lr"],
            "lamda": best["lamda"],
            "val_gauc": best["val_gauc"],
            "val_auc": best["val_auc"],
            "ties_within_1e-6": len(tied),
        }
    out["rule"] = (
        "max train-19 GAUC of the train-18 checkpoint; ties < 1e-6: higher AUC, then closest to lr 0.003, lambda 0.3"
    )
    out["grid"] = rows  # every planned job, checked complete above
    dest = groups.OUT / "config_choice.json"
    if dest.exists():
        raise CheckFailed(f"{dest} exists: the settings are chosen once")
    groups.OUT.mkdir(parents=True, exist_ok=True)
    out[evaluation.WEIGHTED] = {"lr": out["MoAE"]["lr"], "lamda": out["MoAE"]["lamda"], "derived_from": "MoAE"}
    dest.write_text(json.dumps({k: out[k] for k in ("MoAE", "BASE", evaluation.WEIGHTED)}, indent=2))
    (groups.OUT / "config_choice_record.json").write_text(json.dumps(out, indent=2))
    return out


# ------------------------------------------------------------------ freeze
def checkpoints(choice: dict, after: str) -> dict[str, str]:
    """SHA-256 of the checkpoint after `after` of every chosen run, from the run records copied from the pod."""
    out = {}
    for model, c in choice.items():
        for s in evaluation.SEEDS:
            rid = evaluation.run_id(model, c["lr"], c["lamda"], s)
            rec = json.loads((evaluation.RUNS / rid / "run.json").read_text())
            out[rid] = rec["checkpoints"][after]["sha256"]
    return out


def official_test_sha() -> str:
    side = evaluation.TEST_PATH.with_name(evaluation.TEST_PATH.name + ".sha256")
    return side.read_text().split()[0]


def hashed_files() -> dict[str, str]:
    files = [
        groups.OUT / f
        for f in (
            "groups.json",
            "group_lists.json",
            "train_stats_ad_feat1.parquet",
            "config_choice.json",
            "q2/selection.json",
        )
    ]
    files += sorted((groups.OUT / "q2" / "calibrators").glob("*.npz"))
    return {str(p.relative_to(groups.OUT)): sha256(p) for p in files}


def freeze() -> dict:
    FROZEN.mkdir(parents=True, exist_ok=True)
    if (FROZEN / "freeze_q2.json").exists() or (FROZEN / "freeze_q1.json").exists():
        raise CheckFailed("a freeze manifest exists; it is never rewritten")
    choice = json.loads((groups.OUT / "config_choice.json").read_text())
    sel = json.loads((groups.OUT / "q2" / "selection.json").read_text())
    if sel["choice"] != choice:
        raise CheckFailed("the Q2 selection was computed for other settings")
    if set(choice) != {"MoAE", "BASE", evaluation.WEIGHTED}:
        raise CheckFailed(f"the freeze needs MoAE, BASE and {evaluation.WEIGHTED}; got {sorted(choice)}")
    ck17 = checkpoints(choice, "train-17.parquet")  # Q2 and Q4: nine checkpoints in one claim
    ck19 = checkpoints({k: choice[k] for k in ("MoAE", "BASE")}, "train-19.parquet")  # Q1 reproduces the paper's two
    if len(set(ck17.values())) != 9 or len(set(ck19.values())) != 6:
        raise CheckFailed(
            f"expected 9 distinct train-17 and 6 train-19 checkpoints, got {len(set(ck17.values()))} "
            f"and {len(set(ck19.values()))}"
        )
    for rid, s in ck17.items():
        if sel["inputs"][rid]["checkpoint_sha256"] != s:
            raise CheckFailed(f"{rid}: the V1/V2 predictions were not made with its train-17 checkpoint")
    test_sha = official_test_sha()
    common = {
        "official_test_sha256": test_sha,
        "files_sha256": hashed_files(),
        "choice": choice,
        "groups_sha256": sel["groups_sha256"],
        "design": "docs/DESIGN.md §18 (amendments 1-2)",
    }
    lam = sel["lambda_chosen"]
    q2 = common | {
        "question": "Q2",
        "checkpoints_sha256": sorted(ck17.values()),
        "checkpoints": ck17,
        "primary": {
            "comparison": f"MoAE (iii, lambda {lam['MoAE']['iii']['main']:g}) - MoAE (ii)",
            "metric": "G1 macro LogLoss on the official test, seed-paired user bootstrap "
            f"({evaluation.BOOT['reps']} reps, seed {evaluation.BOOT['seed']}), models and calibrators fixed",
            "improvement_confirmed": "difference < 0, 95% CI upper < 0, all three seed pairs < 0",
        },
        "secondary": [
            "BASE: iii - ii",
            "MoAE/BASE: iii0 - ii, iii0 - iii, iii (aggregate rule) - ii",
            "Q3: (MoAE - BASE) on G1 minus on C, after (ii) and raw, no verdict",
        ],
        "q4": {
            "comparison": f"{evaluation.WEIGHTED} (iii) - MoAE (iii), each with its own lambda chosen on V2",
            "metric": "G1 macro LogLoss on the official test, same bootstrap and verdict rule as Q2",
            "guards": "hold the method if the all-click LogLoss difference has 95% CI lower > 0 or the G1 Poisson "
            "deviance gets worse",
            "design": "docs/DESIGN.md §18.10 amendment 3",
        },
        "lambda_chosen": lam,
    }
    q1 = common | {
        "question": "Q1",
        "checkpoints_sha256": sorted(ck19.values()),
        "checkpoints": ck19,
        "paper_table4": PAPER,
        "tolerance": TOL,
        "rule": "per model, AUC and GAUC of the 3-seed mean of the train-19 checkpoints vs Table 4: "
        "<= 0.002 reproduced, <= 0.005 close, otherwise not reproduced",
    }
    for name, m in (("q1", q1), ("q2", q2)):
        (FROZEN / f"freeze_{name}.json").write_text(json.dumps(m, indent=2))
    return {"q1": q1, "q2": q2}


def verify_freeze(name: str) -> dict:
    m = json.loads((FROZEN / f"freeze_{name}.json").read_text())
    now = hashed_files()
    bad = sorted(k for k in set(now) | set(m["files_sha256"]) if now.get(k) != m["files_sha256"].get(k))
    if bad:
        raise CheckFailed(f"frozen files changed since freeze: {bad[:10]}")
    return m


# ------------------------------------------------------------------ evaluation shared by the dry run and the test
def q2_values(name: str, manifest: dict) -> dict:
    """Every planned Q2/Q3 value on file `name` with the frozen calibrators (the same code for V2 and the test)."""
    codes = evaluation.token_codes()
    sets = evaluation.group_sets(codes)
    g, y = evaluation.file_columns(name, codes)
    choice = manifest["choice"]  # the chosen lambdas are inside the saved calibrators
    cal, users = {}, None
    for model, c in choice.items():
        for s in evaluation.SEEDS:
            rid = evaluation.run_id(model, c["lr"], c["lamda"], s)
            pr = evaluation.predictions(rid, name, y)
            if pr["checkpoint_sha256"] != manifest["checkpoints"][rid]:
                raise CheckFailed(f"{rid} {name}: predictions of another checkpoint")
            users = pr["user"] if users is None else users
            if not np.array_equal(users, pr["user"]):
                raise CheckFailed("users differ between runs")
            q = {"raw": pr["p"]}
            for meth in evaluation.METHODS:
                for rule in ("main", "aggregate"):
                    q["ii"], q[f"{meth}_{rule}"] = evaluation.load_calibrated(rid, meth, rule, pr["p"], g)
            cal[rid] = q
    rows = {f"{rid}|{k}": evaluation.evaluate(y, v, g, sets, users) for rid, q in cal.items() for k, v in q.items()}

    def seeds(model, k):
        c = choice[model]
        return [cal[evaluation.run_id(model, c["lr"], c["lamda"], s)][k] for s in evaluation.SEEDS]

    m1 = np.isin(g, sets["G1"])

    def boot(model, a, b):
        return metrics.paired_bootstrap_macro_diff(
            y[m1],
            [x[m1] for x in seeds(model, a)],
            [x[m1] for x in seeds(model, b)],
            g[m1],
            users[m1],
            **evaluation.BOOT,
        ) | {"compared": f"{model}: {a} - {b}"}

    out = {}
    for model in choice:
        out[f"{model}: iii - ii"] = boot(model, "iii_main", "ii")
        out[f"{model}: iii0 - ii"] = boot(model, "iii0_main", "ii")
        out[f"{model}: iii0 - iii"] = boot(model, "iii0_main", "iii_main")
        out[f"{model}: iii (aggregate rule) - ii"] = boot(model, "iii_aggregate", "ii")
    for k in ("ii", "raw"):
        out[f"Q3 {k}: (MoAE - BASE) on G1 minus on C"] = evaluation.did_bootstrap(
            y, seeds("MoAE", k), seeds("BASE", k), g, users, sets["G1"], sets["C"], **evaluation.BOOT
        )
        out[f"Q3 click {k}: (MoAE - BASE) on G1 minus on C"] = evaluation.did_click_bootstrap(
            y, seeds("MoAE", k), seeds("BASE", k), g, users, sets["G1"], sets["C"], **evaluation.BOOT
        )
    if evaluation.WEIGHTED in choice:
        out |= evaluation.q4_comparisons(y, g, users, sets, seeds, "iii_main", "iii_main")
    return {"rows": rows, "bootstrap": out}


def hold_flags(b: dict) -> dict:
    """Pre-set reasons to hold back the weighted training even when it improves G1."""
    click, pois = (
        b["Q4 guard: weighted iii - plain iii, all clicks"],
        b["Q4 guard: weighted iii - plain iii, G1 Poisson deviance"],
    )
    return {"all_click_logloss_worse": bool(click["ci95"][0] > 0), "g1_poisson_worse": bool(pois["diff"] > 0)}


def verdict(b: dict) -> dict:
    ok = b["diff"] < 0 and b["ci95"][1] < 0 and all(d < 0 for d in b["per_pair_diff"])
    return {
        "improvement_confirmed": bool(ok),
        "conditions": {
            "diff<0": b["diff"] < 0,
            "ci_upper<0": b["ci95"][1] < 0,
            "all_pairs<0": all(d < 0 for d in b["per_pair_diff"]),
        },
    }


def dryrun() -> dict:
    m = verify_freeze("q2")
    sel = json.loads((groups.OUT / "q2" / "selection.json").read_text())
    new = q2_values(evaluation.V2, m)
    lam, n = m["lambda_chosen"], 0
    for key, row in new["rows"].items():
        rid, k = key.split("|")
        model = rid.split("_")[0]
        name = (
            k
            if k in ("raw", "ii")
            else f"{k.rsplit('_', 1)[0]}_lam{lam[model][k.rsplit('_', 1)[0]][k.rsplit('_', 1)[1]]:g}"
        )
        rec = sel["v2_rows"][f"{rid}|{name}"]
        for part in ("G1", "C"):
            for metric in ("macro_logloss", "click_logloss", "poisson_mean"):
                require_close(f"dry run {key} {part} {metric}", abs(row[part][metric] - rec[part][metric]), DRYRUN_TOL)
                n += 1
    extra = set(new["bootstrap"]) - set(sel["v2_bootstrap"])
    if extra != evaluation.ADDED_AFTER_SELECTION or set(sel["v2_bootstrap"]) - set(new["bootstrap"]):
        raise CheckFailed(
            f"dry run: values differ from the selection record beyond the ones added after selection: {sorted(extra)}"
        )
    added = {k: new["bootstrap"][k] for k in sorted(extra)}  # recorded, not reproduced
    for key, b in new["bootstrap"].items():
        if key in extra:
            continue
        rec = sel["v2_bootstrap"][key]
        for f in ("diff", "did"):
            if f in b:
                require_close(f"dry run {key} {f}", abs(b[f] - rec[f]), DRYRUN_TOL)
                n += 1
        if "ci95" in b:  # the Poisson guard is a point difference
            require_close(f"dry run {key} CI", max(abs(x - z) for x, z in zip(b["ci95"], rec["ci95"])), DRYRUN_TOL)
            n += 1
        if "per_pair_diff" in b:  # the verdict needs every seed pair below 0, not just the mean
            if len(b["per_pair_diff"]) != len(rec["per_pair_diff"]):
                raise CheckFailed(
                    f"dry run {key}: {len(b['per_pair_diff'])} seed pairs, recorded {len(rec['per_pair_diff'])}"
                )
            for i, (x, z) in enumerate(zip(b["per_pair_diff"], rec["per_pair_diff"])):
                require_close(f"dry run {key} seed pair {i}", abs(x - z), DRYRUN_TOL)
                n += 1
    judged = {"Q2": "MoAE: iii - ii", "Q4": "Q4: weighted iii - plain iii"}
    for q, key in judged.items():
        if verdict(new["bootstrap"][key]) != verdict(sel["v2_bootstrap"][key]):
            raise CheckFailed(f"dry run: the {q} verdict differs from the recorded V2 values")
    holds = [hold_flags(x) for x in (new["bootstrap"], sel["v2_bootstrap"])]
    if holds[0] != holds[1]:
        raise CheckFailed(f"dry run: the Q4 hold conditions differ: {holds}")
    git = git_state()
    out = {
        "dryrun_passed": True,
        "values_reproduced": n,
        "freeze_sha256": sha256(FROZEN / "freeze_q2.json"),
        "v2_q4_verdict_code_run": verdict(new["bootstrap"]["Q4: weighted iii - plain iii"]) | {"hold": holds[0]},
        "v2_values_added_after_selection": added,
        "v2_overall_gauc": {k: r["overall"]["gauc"] for k, r in new["rows"].items()},
        "code_fingerprint": code_fingerprint(),
        "git": git,
        "v2_verdict_code_run": verdict(new["bootstrap"]["MoAE: iii - ii"]) | {"note": "V2 = selection data"},
    }
    (FROZEN / "dryrun_q2.json").write_text(json.dumps(out, indent=2))
    return out


# ------------------------------------------------------------------ the single evaluation of each question
def claim(name: str) -> Path:
    lock = FROZEN / f"evaluation_{name}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise CheckFailed(f"{lock} exists: {name} was evaluated once already") from None
    os.write(fd, json.dumps({"pid": os.getpid()}).encode())
    os.close(fd)
    return lock


def guard(name: str) -> tuple[dict, dict]:
    m = verify_freeze(name)
    git = git_state()
    if git["dirty_files"]:
        raise CheckFailed(f"uncommitted changes: {git['dirty_files'][:5]}")
    require_commit(git, "the test")
    if name == "q2":
        dry = json.loads((FROZEN / "dryrun_q2.json").read_text()) if (FROZEN / "dryrun_q2.json").exists() else {}
        if not dry.get("dryrun_passed") or dry.get("freeze_sha256") != sha256(FROZEN / "freeze_q2.json"):
            raise CheckFailed("the V2 dry run must pass for this freeze before the test")
        if dry.get("code_fingerprint") != code_fingerprint():
            raise CheckFailed("the dry run used different evaluation code")
        require_commit(dry.get("git", {}), "the dry run")
        if dry.get("git", {}).get("dirty_files"):
            raise CheckFailed("the dry run was run on an uncommitted tree")
    if official_test_sha() != m["official_test_sha256"]:
        raise CheckFailed("the local official test file differs from the frozen one")
    check_test_predictions(name, m)
    return m, git


def check_test_predictions(name: str, m: dict) -> None:
    """Check every test input before the single evaluation is claimed, so a missing file cannot waste it.

    The prediction step must have finished for this manifest, and every frozen checkpoint needs its test .npz and
    .json with matching file, checkpoint and source hashes, the expected fields and the file's labels."""
    status = FROZEN / f"freeze_{name}.official_test_status.json"
    if not status.exists() or json.loads(status.read_text()).get("state") != "done":
        raise CheckFailed(f"the test predictions for {name} are not complete ({status})")
    folder, stem = ("q1", "model_after_train-19") if name == "q1" else ("research", "model_after_train-17")
    _, y = evaluation.file_columns(evaluation.TEST, evaluation.token_codes())
    for rid, ck in m["checkpoints"].items():
        f = evaluation.prediction_file(folder, stem, ck, evaluation.TEST)
        meta_f = f.with_suffix(".json")
        if not (f.exists() and meta_f.exists()):
            raise CheckFailed(f"{rid}: no test predictions at {f}")
        meta = json.loads(meta_f.read_text())
        if (
            meta.get("sha256") != sha256(f)
            or meta.get("checkpoint_sha256") != ck
            or meta.get("source_sha256") != m["official_test_sha256"]
        ):
            raise CheckFailed(f"{rid}: the test predictions do not match their record, checkpoint or the official test")
        z = np.load(f)
        if not {"p_last", "user", "y_last"} <= set(z.files):  # fail now, not after the claim
            raise CheckFailed(f"{rid}: test predictions lack p_last, user or y_last")
        p = z["p_last"]
        if not (p.size == z["user"].size == z["y_last"].size == y.size):
            raise CheckFailed(
                f"{rid}: {p.size} predictions, {z['user'].size} users, {z['y_last'].size} labels, {y.size} rows"
            )
        if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
            raise CheckFailed(f"{rid}: test probabilities outside [0, 1] or not finite")
        if not np.array_equal(z["y_last"].astype(np.int8), y):
            raise CheckFailed(f"{rid}: test labels differ from the official test file read again")
        if name == "q1" and not all(np.isfinite(meta.get(k, np.nan)) for k in ("auc", "gauc")):
            raise CheckFailed(f"{rid}: the Q1 record lacks a finite AUC or GAUC")


def test(name: str) -> dict:
    m, git = guard(name)
    lock = claim(name)
    status = FROZEN / f"evaluation_{name}_status.json"
    status.write_text(json.dumps({"state": "started", "lock": str(lock)}))
    try:
        out = (_test_q2 if name == "q2" else _test_q1)(m) | {
            "git": git,
            "code_fingerprint": code_fingerprint(),
            "freeze_sha256": sha256(FROZEN / f"freeze_{name}.json"),
        }
    except BaseException as e:
        status.write_text(json.dumps({"state": "failed_after_claim", "error": repr(e)[:2000]}))
        raise
    (FROZEN / f"result_{name}.json").write_text(json.dumps(out, indent=2, default=str))
    status.write_text(json.dumps({"state": "done"}))
    return out


def _test_q2(m: dict) -> dict:
    v = q2_values(evaluation.TEST, m)
    b = v["bootstrap"]
    out = {
        "question": "Q2",
        "primary": b["MoAE: iii - ii"],
        "verdict": verdict(b["MoAE: iii - ii"]),
        "secondary": b,
        "rows": v["rows"],
    }
    if "Q4: weighted iii - plain iii" in b:
        out["q4"] = {
            "primary": b["Q4: weighted iii - plain iii"],
            "verdict": verdict(b["Q4: weighted iii - plain iii"]),
            "hold": hold_flags(b),
        }
    return out


def _test_q1(m: dict) -> dict:
    from sklearn.metrics import roc_auc_score

    codes = evaluation.token_codes()
    _, y = evaluation.file_columns(evaluation.TEST, codes)
    per, out = {}, {}
    for model in ("MoAE", "BASE"):
        c = m["choice"][model]
        vals = []
        for s in evaluation.SEEDS:
            rid = evaluation.run_id(model, c["lr"], c["lamda"], s)
            hits = [evaluation.prediction_file("q1", "model_after_train-19", m["checkpoints"][rid], evaluation.TEST)]
            if not hits[0].exists():
                raise CheckFailed(f"{rid}: no Q1 test predictions at {hits[0]}")
            z = np.load(hits[0])
            rec = json.loads(hits[0].with_suffix(".json").read_text())
            if rec["checkpoint_sha256"] != m["checkpoints"][rid] or rec["source_sha256"] != m["official_test_sha256"]:
                raise CheckFailed(f"{rid}: Q1 predictions of another checkpoint or test file")
            if not np.array_equal(z["y_last"].astype(np.int8), y):
                raise CheckFailed(f"{rid}: Q1 labels differ from the test file read again")
            auc, gauc = float(roc_auc_score(y, z["p_last"])), metrics.gauc(y, z["p_last"], z["user"])["gauc"]
            require_close(f"{rid} AUC official vs project", abs(auc - rec["auc"]), 1e-9)
            require_close(f"{rid} GAUC official vs project", abs(gauc - rec["gauc"]), 1e-9)
            vals.append({"seed": s, "auc": auc, "gauc": gauc})
        per[model] = vals
        mean = {k: float(np.mean([v[k] for v in vals])) for k in ("auc", "gauc")}
        out[model] = {
            "mean": mean,
            "sd": {k: float(np.std([v[k] for v in vals], ddof=1)) for k in ("auc", "gauc")},
            "paper": PAPER[model],
            "gap": {k: mean[k] - PAPER[model][k] for k in ("auc", "gauc")},
            "judgement": {
                k: (
                    "reproduced"
                    if abs(mean[k] - PAPER[model][k]) <= TOL["reproduced"]
                    else "close"
                    if abs(mean[k] - PAPER[model][k]) <= TOL["close"]
                    else "not reproduced"
                )
                for k in ("auc", "gauc")
            },
        }
    gap = {k: out["MoAE"]["mean"][k] - out["BASE"]["mean"][k] for k in ("auc", "gauc")}
    return {
        "question": "Q1",
        "per_seed": per,
        "summary": out,
        "MoAE_minus_BASE": gap,
        "paper_MoAE_minus_BASE": {k: PAPER["MoAE"][k] - PAPER["BASE"][k] for k in ("auc", "gauc")},
    }


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "choose":
        r = choose(Path(sys.argv[2]))
        print(json.dumps({k: r[k] for k in ("MoAE", "BASE", "rule")}, indent=2))
    elif cmd == "freeze":
        print(json.dumps({k: v["checkpoints_sha256"] for k, v in freeze().items()}, indent=2))
    elif cmd == "dryrun":
        print(json.dumps(dryrun(), indent=2, default=str))
    elif cmd == "test" and sys.argv[2:] in (["q1"], ["q2"]):
        r = test(sys.argv[2])
        print(json.dumps(r.get("summary") or {"primary": r["primary"], "verdict": r["verdict"]}, indent=2, default=str))
    else:
        raise SystemExit(__doc__)
