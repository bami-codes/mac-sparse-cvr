"""Study E job queue on a multi-GPU pod (docs/DESIGN.md §18.10, amendment 2): one training run per GPU at a time.

    python scripts/pod_queue.py grid  > runs/e_runs/jobs_grid.json          # the paper's search space, seed 2026
    python scripts/pod_queue.py seeds --moae "<lr> <lamda>" --base "<lr>" > runs/e_runs/jobs_seeds.json
    python scripts/pod_queue.py run runs/e_runs/jobs_grid.json [--gpus 4]   # resumable: run it again after a stop
    python scripts/pod_queue.py summary runs/e_runs/jobs_grid.json          # status and validation GAUC per job

Each job trains with cvr.pymal_run on the official 20 files (checkpoints after train-17, 18, 19) and then scores
the train-18 checkpoint on train-19 (the paper's validation: the day before the test day). The official test file
is never touched here. A job is skipped when its validation record exists, a stopped training is continued with
--resume, and a diverged run (non-finite loss) is recorded and not validated.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RUNS = Path("runs/e_runs")
LRS = [round(x * f, 8) for f in (0.1, 1.0, 10.0) for x in (0.0030, 0.0035, 0.0040)]  # the paper's grid (§6.1.1)
LAMDAS = [0.1, 0.2, 0.3, 0.4]
MOAE_AUX = ["first", "mta", "linear", "cartesian"]  # Table 4 / Table 5: first, linear, DDA (mta) and CAT


def job(model: str, lr: float, lamda: float, seed: int) -> dict:
    jid = f"{model}_lr{lr:g}_lam{lamda:g}_s{seed}"
    args = ["--model", model, "--lr", repr(lr), "--lamda", repr(lamda), "--seed", str(seed)]
    if model == "MoAE":
        args += ["--aux_views", *MOAE_AUX]
    return {"id": jid, "model": model, "lr": lr, "lamda": lamda, "seed": seed, "train_args": args}


WEIGHTED = "MoAE-g1w2"  # amendment 3 (Q4)


def weighted(lr: float, lamda: float) -> list[dict]:
    """The chosen MoAE retrained on train-00..17 with the G1 clicks' main loss weighted 2x; only the train-17
    checkpoint is kept and no validation run is made (the control is the chosen run's own train-17 checkpoint)."""
    out = []
    for seed in (2026, 2027, 2028):
        j = job("MoAE", lr, lamda, seed)
        j["id"] = j["id"].replace("MoAE_", f"{WEIGHTED}_", 1)
        j["model"] = WEIGHTED
        j["train_args"] += ["--g1_weight", "2", "--weight_groups", "runs/e_study/group_lists.json",
                            "--train_files", *[f"train-{i:02d}.parquet" for i in range(18)], "--ckpt_after", "train-17.parquet"]
        j["validate"] = False
        out.append(j)
    return out


def grid() -> list[dict]:
    jobs = [job("MoAE", lr, lam, 2026) for lr in LRS for lam in LAMDAS]
    jobs += [job("BASE", lr, 0.3, 2026) for lr in LRS]  # BASE has no auxiliary target: lambda does not enter its loss
    return jobs


def state(j: dict) -> str:
    d = RUNS / j["id"]
    if (d / "run.json").exists():
        rec = json.loads((d / "run.json").read_text())
        if rec.get("diverged"):
            return "diverged"
        if not j.get("validate", True):
            return "done"
        return "done" if list((d / "val").rglob("train-19.json")) else "validate"
    if (d / "progress.json").exists():
        return "resume"
    return "train"


def command(j: dict, st: str, extra: list[str]) -> list[str]:
    d = RUNS / j["id"]
    py = [sys.executable, "-m", "cvr.pymal_run"]
    if st == "train":
        return py + ["train", "--out", str(d), *extra, *j["train_args"]]
    if st == "resume":
        return py + ["train", "--resume", str(d), *extra]
    return py + ["predict", "--out", str(d / "val"), "--ckpt", str(d / "model_after_train-18.pt"), *extra,
                 "train-19.parquet"]


def run(jobs: list[dict], gpus: int, extra: list[str] | None = None) -> None:
    extra = extra or []
    RUNS.mkdir(parents=True, exist_ok=True)
    busy: dict[int, tuple[subprocess.Popen, dict, object]] = {}
    pending = [j for j in jobs if state(j) not in ("done", "diverged")]
    while pending or busy:
        for g in [g for g in range(gpus) if g not in busy]:
            if not pending:
                break
            j = pending.pop(0)
            st = state(j)
            (RUNS / j["id"]).mkdir(parents=True, exist_ok=True)
            log = open(RUNS / f"{j['id']}.log", "a")
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {st} on GPU {g}\n")
            log.flush()
            env = os.environ | {"CUDA_VISIBLE_DEVICES": str(g)}
            if st == "train" and (RUNS / j["id"]).exists() and not any((RUNS / j["id"]).iterdir()):
                (RUNS / j["id"]).rmdir()  # cvr.pymal_run refuses an existing run folder
            busy[g] = (subprocess.Popen(command(j, st, extra), stdout=log, stderr=subprocess.STDOUT, env=env), j, log)
            print(f"{time.strftime('%H:%M:%S')} GPU {g}: {st} {j['id']}", flush=True)
        time.sleep(10)
        for g, (p, j, log) in list(busy.items()):
            if p.poll() is None:
                continue
            log.close()
            del busy[g]
            st = state(j)
            print(f"{time.strftime('%H:%M:%S')} GPU {g}: {j['id']} exit {p.returncode} -> {st}", flush=True)
            if p.returncode == 0 and st in ("validate", "resume"):
                pending.insert(0, j)  # next stage of the same job
            elif p.returncode != 0:
                print(f"   failed: see {RUNS / (j['id'] + '.log')}", flush=True)


def summary(jobs: list[dict]) -> list[dict]:
    rows = []
    for j in jobs:
        d, st = RUNS / j["id"], state(j)
        r = {k: j[k] for k in ("id", "model", "lr", "lamda", "seed")} | {"state": st}
        if (d / "run.json").exists():
            rec = json.loads((d / "run.json").read_text())
            r["train_seconds"] = round(sum(f["seconds"] for f in rec["files"].values()))
            r["diverged"] = rec.get("diverged")
        v = sorted((d / "val").rglob("train-19.json"))
        if v:
            m = json.loads(v[0].read_text())
            r |= {"val_gauc": m.get("gauc"), "val_auc": m.get("auc")}
        rows.append(r)
    return rows


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("grid")
    p = sub.add_parser("seeds")
    p.add_argument("--moae", required=True, help="'<lr> <lamda>' chosen on train-19")
    p.add_argument("--base", required=True, help="'<lr>' chosen on train-19")
    p.add_argument("--with-weighted", action="store_true", help="add amendment 3's three weighted MoAE runs")
    for name in ("run", "summary"):
        p = sub.add_parser(name)
        p.add_argument("jobs", type=Path)
        p.add_argument("--gpus", type=int, default=4)
        p.add_argument("--device", default="cuda")
        p.add_argument("--num_workers", type=int, default=8)
    a = ap.parse_args(argv)
    if a.cmd == "grid":
        print(json.dumps(grid(), indent=1))
    elif a.cmd == "seeds":
        lr, lam = (float(x) for x in a.moae.split())
        jobs = ([job("MoAE", lr, lam, s) for s in (2027, 2028)] + [job("BASE", float(a.base), 0.3, s) for s in (2027, 2028)]
                + (weighted(lr, lam) if a.with_weighted else []))
        print(json.dumps(jobs, indent=1))
    elif a.cmd == "run":
        run(json.loads(a.jobs.read_text()), a.gpus, ["--device", a.device, "--num_workers", str(a.num_workers)])
    else:
        for r in summary(json.loads(a.jobs.read_text())):
            print(json.dumps(r))


if __name__ == "__main__":
    main(sys.argv[1:])
