"""Study C: freeze a development run, dry-run the evaluation on V2, then evaluate train-13 + train-14 ONCE.

    uv run python -m cvr.final_eval_c expect runs/c_dev_train08_10/<stamp>   # official holdout size / SHA-256 only
    uv run python -m cvr.final_eval_c freeze runs/c_dev_train08_10/<stamp>
    uv run python -m cvr.final_eval_c dryrun runs/c_dev_train08_10/<stamp>
    uv run python -m cvr.final_eval_c test   runs/c_dev_train08_10/<stamp> data/raw/mac/train/train-13.parquet \
                                                                            data/raw/mac/train/train-14.parquet

freeze  requires the independent re-selection (diagnostics_calib.json) and the raw-row serving check with every
        seed's p99 (latency_c.json); recomputes the domain and G1 statistics from the run's training arrays and
        checks them against selection.json; writes freeze.json with the SHA-256 of every model, calibrator,
        vocabulary, selection, latency and diagnostic file, the primary comparison, the verdict rules, the
        secondary comparisons and the holdout files. freeze.json is never overwritten.
dryrun  runs the test code on V2 and requires the saved predictions, every recorded V2 metric and the primary
        bootstrap to be reproduced within 1e-9 (hard check).
expect  records the official size and SHA-256 of train-13 / train-14 from the Hugging Face API (metadata only, no
        data is downloaded), so the test can refuse any other file under a holdout name.
test    BEFORE claiming the single run: every frozen hash, a dry run recorded for this freeze, this evaluation code
        and a clean tree, the frozen holdout names, and each file's size and official SHA-256 (a failure here can
        be retried). Then ONE process claims the run with an exclusive lock file; from then on any failure is
        recorded and the run counts as used. Both files are evaluated together (a user seen on both days is one
        bootstrap cluster) plus each file's direction; nothing is selected or tuned.

Networks are scored in a PyTorch-only subprocess (cvr/final_eval_c_nn.py); this process loads LightGBM.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import polars as pl

from . import domain_eval as D
from . import groups as G
from .artifacts import load_gbdt
from .checks import (NN_SINGLE_VS_BATCH_TOL, NUM_TOL, PRED_TOL, CheckFailed, require_close, require_commit,
                     require_finite_probs)
from .domain_models import load_domain_calibrators, load_gbdt_offset
from .features import featurize, read_raw
from .mddata import DOM_COL, encode_md, load_md, save_md
from .schema import CAT_INPUTS, DOMAIN_KEY, GROUP_KEY

DRYRUN_TOL = 1e-9
FAMILIES = ["G-int", "G-shared", "N-int", "N-shared", "N-head", "N-star", "N-star-noPN"]
HOLDOUT = ["train-13.parquet", "train-14.parquet"]
HF_TREE = "https://huggingface.co/api/datasets/alimamaTech/MAC/tree/main/train"
LATENCY_FAMILIES = ["G-int", "G-shared", "N-int", "N-shared", "N-head", "N-star"]
LATENCY_PROTOCOL_VERSION, LATENCY_REQUESTS, ACCEPT_OTHER_CPU = 2, 5000, 400.0  # same values as scripts/latency_c.py
LATENCY_MAX_ATTEMPTS = 3  # scripts/latency_c.MAX_ATTEMPTS: a family is measured at most this many times in all
COMPARED_METRICS = [("small", "macro_logloss"), ("small", "click_weighted_logloss"), ("small", "within_domain_auc_macro"),
                    ("small", "poisson_deviance_mean"), ("large", "click_weighted_logloss"),
                    ("overall_nonmissing", "logloss"), ("overall_nonmissing", "auc"), ("G1_macro_logloss",)]


def git_state() -> dict:
    """Commit and uncommitted files (local copy: cvr.pipeline imports LightGBM at module level)."""
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty_files": run("status", "--porcelain").splitlines()}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hashed_files(run: Path) -> dict[str, str]:
    files = [p for d in ("models", "vocab") for p in sorted((run / d).rglob("*")) if p.is_file()]
    files += [run / f for f in ("selection.json", "results.json", "latency_c.json", "diagnostics_calib.json",
                                "diagnostics_star.json", "frozen_domain_stats.parquet", "frozen_g1_stats.parquet",
                                "config.toml", "holdout_expected.json")]
    return {str(p.relative_to(run)): sha256(p) for p in files}


def code_fingerprint() -> str:
    """SHA-256 over every module of the cvr package: the dry run and the test must run the same evaluation code."""
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


# ------------------------------------------------------------------ expected holdout files (metadata only)
def expect(run: Path) -> dict:
    out = run / "holdout_expected.json"
    if out.exists():
        raise CheckFailed(f"{out} already exists")
    import urllib.request
    from datetime import datetime, timezone

    with urllib.request.urlopen(HF_TREE, timeout=60) as r:
        items = json.loads(r.read())
    files = {}
    for name in HOLDOUT:
        hit = [i for i in items if i.get("path") == f"train/{name}"]
        if len(hit) != 1 or "lfs" not in hit[0]:
            raise CheckFailed(f"{name} not found once in {HF_TREE}")
        files[name] = {"size": int(hit[0]["size"]), "sha256": hit[0]["lfs"]["oid"]}
    root = Path(json.loads((run / "results.json").read_text())["meta"]["data"]["v2"][0]["file"]).parent
    rec = {"source": HF_TREE, "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "files": files, "local_copy_present_when_recorded": {n: (root / n).exists() for n in HOLDOUT}}
    out.write_text(json.dumps(rec, indent=2))
    return rec


def latency_accepted(start: dict, end: dict) -> bool:
    return (not start["other_repo_python"] and not end["other_repo_python"]
            and start["other_cpu_percent"] <= ACCEPT_OTHER_CPU and end["other_cpu_percent"] <= ACCEPT_OTHER_CPU)


def check_latency_provenance(report: dict, run: Path, sel: dict) -> None:
    """The latency report must come from this run's selected models and calibrators (a smoke run's report, say,
    with lambda 100 instead of 1), and the kept attempt must be the first one that meets the acceptance rule, with
    every earlier attempt rejected and at most LATENCY_MAX_ATTEMPTS in all."""
    if Path(report["run"]).resolve() != run.resolve():
        raise CheckFailed(f"latency report is for {report['run']}, not {run}")
    from .serving_c import model_path
    for fam in LATENCY_FAMILIES:
        f = report["families"][fam]
        lam = float(sel["lambda_iiid"][fam])
        for s in D.SEEDS:
            info = f["seeds"][str(s)]["server"]
            want = {"model": model_path(run, fam, s).resolve(), "calibrator": f"calib_{fam}_s{s}", "lambda": lam,
                    "group_key": DOMAIN_KEY, "calibration": "iiid"}
            got = {"model": Path(info["model"]).resolve(), "calibrator": info["calibrator"],
                   "lambda": float(info["lambda"]), "group_key": info["group_key"], "calibration": info["calibration"]}
            if got != want:
                raise CheckFailed(f"latency of {fam} seed {s} was measured on {got}, expected {want}")
        attempts = f.get("attempts") or []
        if len(attempts) > LATENCY_MAX_ATTEMPTS:  # more tries would let a measurement be repeated until it is accepted
            raise CheckFailed(f"latency of {fam}: {len(attempts)} attempts, at most {LATENCY_MAX_ATTEMPTS} are allowed")
        flags = [latency_accepted(a["environment_start"], a["environment_end"]) for a in attempts]
        if not attempts or True not in flags or flags.index(True) != len(attempts) - 1:
            raise CheckFailed(f"latency of {fam}: attempts {flags}; the kept one must be the first accepted")
        if attempts[-1]["p99_ms_by_seed"] != f["p99_ms_by_seed"] or attempts[-1]["environment_start"] != f["environment_start"]:
            raise CheckFailed(f"latency of {fam}: the kept attempt differs from the detailed record")


def frozen_p99(lat: dict) -> tuple[dict, dict]:
    """Per family and seed, the p99 (ms) of the uniform-traffic replay, taken from the DETAILED per-seed record.
    Refused unless the record uses the pre-registered protocol (version 2, 5,000 timed requests per seed), passed
    the serving check (codes, numeric features, probabilities) and meets the acceptance rule recomputed from its
    recorded environment. One seed, a negative value, a NaN summary, a numeric-feature error and the excluded
    1,000-request measurement are all refused."""
    p99, strat = {}, {}
    for fam in LATENCY_FAMILIES:
        f = lat[fam]
        proto = f.get("protocol", {})
        if proto.get("version") != LATENCY_PROTOCOL_VERSION or proto.get("requests") != LATENCY_REQUESTS:
            raise CheckFailed(f"latency of {fam} was not measured with protocol v{LATENCY_PROTOCOL_VERSION} "
                              f"({LATENCY_REQUESTS} requests): {proto}")
        st, en = f.get("environment_start"), f.get("environment_end")
        if not st or not en:
            raise CheckFailed(f"latency of {fam} has no recorded environment")
        if not latency_accepted(st, en) or f.get("accepted") is not True:
            raise CheckFailed(f"latency of {fam} does not meet the acceptance rule: start {st}, end {en}")
        if sorted(int(k) for k in f["seeds"]) != sorted(D.SEEDS):
            raise CheckFailed(f"latency of {fam} measured for seeds {sorted(f['seeds'])}, expected {D.SEEDS}")
        tol = PRED_TOL if fam.startswith("G-") else NN_SINGLE_VS_BATCH_TOL
        p99[fam], strat[fam] = {}, {}
        for s in D.SEEDS:
            rec = f["seeds"][str(s)]
            c = rec["consistency"]
            if not (c["codes_mismatch"] == 0 and np.isfinite(c["num_max_abs_diff"]) and c["num_max_abs_diff"] <= NUM_TOL
                    and np.isfinite(c["prediction_max_abs_diff"]) and c["prediction_max_abs_diff"] <= tol
                    and c["tolerances"]["pred"] == tol):
                raise CheckFailed(f"serving check of {fam} seed {s} did not pass: {c}")
            if rec["batch1_traffic"].get("requests") != LATENCY_REQUESTS:
                raise CheckFailed(f"{fam} seed {s}: {rec['batch1_traffic'].get('requests')} timed requests")
            v = float(rec["batch1_traffic"]["p99_ms"])
            summary = float(f["p99_ms_by_seed"][str(s)])
            if not (np.isfinite(v) and v > 0 and np.isfinite(summary) and summary == v):
                raise CheckFailed(f"p99 of {fam} seed {s}: detail {v}, summary {summary}")
            p99[fam][s] = v
            strat[fam][s] = float(rec["batch1_stratified"]["p99_ms"])
    return p99, strat


def train_stats(run: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Domain and ad_feat1 (G1) statistics recomputed from the run's own training arrays."""
    tr = load_md(run / "nn_data" / "train")
    y = np.load(run / "nn_data" / "y_train.npy").astype(np.int8)
    df = pl.DataFrame({DOMAIN_KEY: np.asarray(tr.cat[:, DOM_COL]),
                       GROUP_KEY: np.asarray(tr.cat[:, CAT_INPUTS.index(GROUP_KEY)]), "y": y})
    return D.domain_stats(df), G.train_stats(df)


def family_seeds(fam: str, seeds: list[int]) -> list[str]:
    return [f"{fam}_s{s}" for s in seeds]


# ------------------------------------------------------------------ freeze
def freeze(run: Path) -> dict:
    out = run / "freeze.json"
    if out.exists():
        raise CheckFailed(f"{out} already exists; a frozen run is never re-frozen")
    cfg = tomllib.loads((run / "config.toml").read_text())
    res = json.loads((run / "results.json").read_text())
    sel = json.loads((run / "selection.json").read_text())
    if json.dumps(res["selection"], sort_keys=True) != json.dumps(sel, sort_keys=True):
        raise CheckFailed("selection.json differs from the selection inside results.json")
    if sorted(sel["seeds"]) != sorted(D.SEEDS):
        raise CheckFailed(f"run seeds {sel['seeds']} differ from the pre-registered {D.SEEDS}")
    diag = json.loads((run / "diagnostics_calib.json").read_text())["calib"]
    if diag["candidate_recomputed_exact"] != sel["candidate"] or diag["lambda_recomputed_exact"] != {
            k: float(v) for k, v in sel["lambda_iiid"].items()}:
        raise CheckFailed("the independent re-selection (diagnostics_calib.json) does not match selection.json")
    latency_report = json.loads((run / "latency_c.json").read_text())
    check_latency_provenance(latency_report, run, sel)
    p99, p99_strat = frozen_p99(latency_report["families"])
    exp = json.loads((run / "holdout_expected.json").read_text())["files"]
    if sorted(exp) != sorted(HOLDOUT) or not all(re.fullmatch(r"[0-9a-f]{64}", v["sha256"]) and v["size"] > 0
                                                 for v in exp.values()):
        raise CheckFailed(f"holdout_expected.json must give the size and SHA-256 of {HOLDOUT}")
    dstats, gstats = train_stats(run)
    dsum = D.domain_summary(dstats, cfg["domains"]["small_share"], cfg["domains"]["small_min_clicks"])
    for k in ("small", "tiny", "large"):
        if dsum[k]["codes"] != sel["domains"][k]:
            raise CheckFailed(f"recomputed {k} domains differ from selection.json")
    dstats.write_parquet(run / "frozen_domain_stats.parquet")
    gstats.sort(GROUP_KEY).write_parquet(run / "frozen_g1_stats.parquet")
    cand, ctrl = sel["candidate"], sel["control"]
    best_g = min(["G-int", "G-shared"], key=lambda f: sel_score(res, f, sel))
    best_n = min(["N-int", "N-shared", "N-head", "N-star"], key=lambda f: sel_score(res, f, sel))
    manifest = {
        "run": str(run), "pipeline_git_commit": res["meta"]["git"]["commit"],
        "pipeline_dirty_files": res["meta"]["git"]["dirty_files"], "config": cfg, "seeds": D.SEEDS,
        "selection": sel, "best_on_v2": {"gbdt": best_g, "nn": best_n},
        "domain_definition": dsum["definition"], "domains": {k: dsum[k]["codes"] for k in ("small", "tiny", "large")},
        "g1_definition": cfg["groups"],
        "primary": {"comparison": f"{cand} (iii-d) - {ctrl} (iii-d), lambda chosen on V2 per family",
                    "metric": "small-domain macro LogLoss (equal weight over small domains), lower is better",
                    "uncertainty": f"seed-paired user bootstrap over BOTH test files, {cfg['bootstrap']['reps']} reps, "
                                   f"seed {cfg['bootstrap']['seed']}; models fixed",
                    "improvement_confirmed": "difference < 0, 95% CI upper < 0, and all three seed pairs < 0",
                    "adoption_candidate": f"improvement confirmed and relative improvement >= {D.ADOPT_REL:.1%}, "
                                          f"large-domain difference 95% CI upper <= {D.ADOPT_LARGE}, "
                                          f"max p99 over seeds <= {D.ADOPT_P99_MS} ms"},
        "p99_ms_by_family_seed": p99,
        "p99_ms_stratified_sample": p99_strat,
        "holdout_expected": exp,
        "power_check_before_test": sel["power_check"],
        "secondary_comparisons": ["ladder table (family x calibration)", "N-star - N-shared", "N-head - N-shared",
                                  "N-star-noPN - N-star", f"best NN on V2 ({best_n}) - best GBDT on V2 ({best_g})",
                                  f"{cand}: (iii-d) - (ii)", f"{ctrl}: (iii-d) - (ii)",
                                  "per test file direction of the primary comparison", "leave one small domain out"]
                                 + ([f"{cand} - G-shared"] if cand != "G-shared" else []),
        "holdout_files": HOLDOUT,
        "rule": "the holdout files are evaluated once, together; nothing is selected or tuned on them",
    }
    manifest["files_sha256"] = hashed_files(run)
    out.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def sel_score(res: dict, fam: str, sel: dict) -> float:
    lam = sel["lambda_iiid"][fam]
    return float(np.mean([res["v2_rows"][f"{n}|iiid_lam{lam:g}"]["small"]["macro_logloss"]
                          for n in family_seeds(fam, sel["seeds"])]))


def verify_freeze(run: Path) -> dict:
    manifest = json.loads((run / "freeze.json").read_text())
    now = hashed_files(run)
    bad = sorted(k for k in set(now) | set(manifest["files_sha256"]) if now.get(k) != manifest["files_sha256"].get(k))
    if bad:
        raise CheckFailed(f"frozen files changed since freeze: {bad[:10]}")
    return manifest


# ------------------------------------------------------------------ evaluation
def score(run: Path, df: pl.DataFrame, work: Path) -> dict[str, np.ndarray]:
    """Raw predictions of every saved base model on `df` (featurized with the run's vocabulary)."""
    mdir = run / "models"
    preds = {}
    for s in D.SEEDS:
        preds[f"G-shared_s{s}"] = load_gbdt(mdir / f"G-shared_s{s}").predict(df)
        preds[f"G-int_s{s}"] = load_gbdt_offset(mdir / f"G-int_s{s}").predict(df)
    save_md(encode_md(df), work / "nn_eval")
    subprocess.run([sys.executable, "-m", "cvr.final_eval_c_nn", str(run), str(work / "nn_eval"), str(work / "nn_preds")],
                   check=True)
    for f in sorted((work / "nn_preds").glob("*.npy")):
        preds[f.stem] = np.load(f)
    missing = [n for fam in FAMILIES for n in family_seeds(fam, D.SEEDS) if n not in preds]
    if missing:
        raise CheckFailed(f"missing predictions: {missing}")
    return preds


def guard_test(run: Path, manifest: dict, files: list[Path], git: dict) -> dict[str, str]:
    """Every check that can fail WITHOUT using the single run. Returns the verified SHA-256 of each file."""
    lock, out_dir = run / "final" / "test.lock", run / "final" / "test"
    if lock.exists() or out_dir.exists():
        raise CheckFailed(f"{lock} or {out_dir} exists: the holdout files are evaluated only once")
    if git["dirty_files"]:
        raise CheckFailed(f"uncommitted changes; commit the evaluation code before the test: {git['dirty_files'][:5]}")
    require_commit(git, "the test")
    dry_path = run / "final" / "dryrun" / "results.json"
    dry = json.loads(dry_path.read_text()) if dry_path.exists() else {}
    if not dry.get("dryrun_passed"):
        raise CheckFailed("the V2 dry run must pass before the test")
    if dry.get("freeze_sha256") != sha256(run / "freeze.json"):
        raise CheckFailed("the dry run was recorded for a different freeze.json")
    if dry.get("code_fingerprint") != code_fingerprint():
        raise CheckFailed("the dry run used different evaluation code (cvr/*.py changed since the dry run)")
    if dry.get("evaluation_git", {}).get("dirty_files"):
        raise CheckFailed("the dry run was run on an uncommitted tree")
    require_commit(dry.get("evaluation_git", {}), "the dry run")
    v2_rec = {Path(x["file"]).name: x["sha256"] for x in
              json.loads((run / "results.json").read_text())["meta"]["data"]["v2"]}  # frozen file
    if dry.get("files_sha256") != v2_rec or None in v2_rec.values():
        raise CheckFailed(f"the dry run's V2 files {dry.get('files_sha256')} differ from the development V2 {v2_rec}")
    if sorted(f.name for f in files) != sorted(manifest["holdout_files"]):
        raise CheckFailed(f"test files {[f.name for f in files]} differ from the frozen holdout {manifest['holdout_files']}")
    verified = {}
    for f in files:
        exp = manifest["holdout_expected"][f.name]
        if not f.is_file():
            raise CheckFailed(f"{f} does not exist")
        if f.stat().st_size != exp["size"]:
            raise CheckFailed(f"{f} has {f.stat().st_size} bytes, the official file has {exp['size']}")
        h = sha256(f)
        if h != exp["sha256"]:
            raise CheckFailed(f"{f} SHA-256 {h} differs from the official {exp['sha256']}")
        verified[f.name] = h
    return verified


def claim_test(run: Path) -> Path:
    """Exactly one process gets the single run: O_CREAT | O_EXCL on the lock file is atomic."""
    lock = run / "final" / "test.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise CheckFailed(f"{lock} exists: another process holds or used the single run") from None
    from datetime import datetime, timezone
    os.write(fd, json.dumps({"pid": os.getpid(), "claimed_utc": datetime.now(timezone.utc).isoformat()}).encode())
    os.close(fd)
    return lock


def evaluate(run: Path, files: list[Path], mode: str) -> dict:
    manifest = verify_freeze(run)
    git = git_state()
    if mode != "test":
        return _evaluate(run, manifest, files, mode, git, None)
    verified = guard_test(run, manifest, files, git)
    lock = claim_test(run)
    status = run / "final" / "test_status.json"
    status.write_text(json.dumps({"state": "started", "lock": str(lock), "files_sha256": verified}))
    try:
        out = _evaluate(run, manifest, files, mode, git, verified)
    except BaseException as e:  # the run is used: record the failure, keep the lock
        status.write_text(json.dumps({"state": "failed_after_claim", "error": repr(e)[:2000], "files_sha256": verified}))
        raise
    status.write_text(json.dumps({"state": "done", "files_sha256": verified}))
    return out


def _evaluate(run: Path, manifest: dict, files: list[Path], mode: str, git: dict, verified: dict | None) -> dict:
    cfg, sel, seeds = manifest["config"], manifest["selection"], manifest["seeds"]
    out_dir = run / "final" / mode
    if mode == "dryrun":
        res_meta = json.loads((run / "results.json").read_text())["meta"]
        recorded = [x["sha256"] for x in res_meta["data"]["v2"]]
        now = [sha256(f) for f in files]
        if None not in recorded and recorded != now:
            raise CheckFailed(f"V2 file hashes {now} differ from the development run's {recorded}")
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = {p.stem: pl.read_parquet(p) for p in (run / "vocab").glob("*.parquet")}
    smoke = cfg["data"]["mode"] == "smoke"
    parts, file_of = [], []
    for k, f in enumerate(files):
        b = tuple(cfg["data"]["smoke_v2_buckets"]) if smoke and mode == "dryrun" else None
        u = cfg["data"].get("smoke_user_pct") if smoke and mode == "dryrun" else None
        part = featurize(read_raw(f, b, u), vocab)
        parts.append(part)
        file_of.append(np.full(part.height, k))
    df = pl.concat(parts) if len(parts) > 1 else parts[0]
    file_of = np.concatenate(file_of)
    dstats = pl.read_parquet(run / "frozen_domain_stats.parquet")
    gstats = pl.read_parquet(run / "frozen_g1_stats.parquet")
    dd = manifest["domain_definition"]
    cls = D.domain_class(df, dstats, dd["small_share"], dd["small_min_clicks"])
    dom, y, users = df[DOMAIN_KEY].to_numpy(), df["y"].to_numpy(), df["user_key"].to_numpy()
    nonmiss = df["is_missing"].to_numpy() == 0
    gc = manifest["g1_definition"]
    g1 = G.classify(df, gstats, gc["n_min"], gc["c_min"], gc["c_dense"])["group_class"].to_numpy() == "G1"
    g1_groups = df[GROUP_KEY].to_numpy()
    preds = score(run, df, out_dir)

    lam = {f: float(v) for f, v in sel["lambda_iiid"].items()}
    cal, rows = {}, {}
    for fam in FAMILIES:
        for name in family_seeds(fam, seeds):
            p = preds[name]
            glob, gs = load_domain_calibrators(run / "models" / f"calib_{name}")
            if float(gs.lam) != lam[fam]:
                raise CheckFailed(f"saved calibrator of {name} has lambda {gs.lam}, frozen {lam[fam]}")
            cal[name] = {"raw": p, "ii": glob.predict(p), "iiid": gs.predict(p, dom)}
            for c, q in cal[name].items():
                require_finite_probs(f"{name}|{c}", q)
                key = f"{name}|{c}" if c != "iiid" else f"{name}|iiid_lam{lam[fam]:g}"
                rows[key] = D.report(y, q, dom, cls, nonmiss, g1, g1_groups)

    def calibrated(fam: str, c: str = "iiid") -> list[np.ndarray]:
        return [cal[n][c] for n in family_seeds(fam, seeds)]

    bs = cfg["bootstrap"]

    def boot(a: str, b: str, ca: str = "iiid", cb: str = "iiid", mask: np.ndarray | None = None) -> dict:
        m = np.ones(y.size, bool) if mask is None else mask
        out = D.paired_bootstrap(y[m], [x[m] for x in calibrated(a, ca)], [x[m] for x in calibrated(b, cb)], dom[m],
                                 cls[m], users[m], bs["reps"], bs["seed"], pair_seeds=seeds)
        out["compared"] = f"{a} ({ca}) - {b} ({cb})"
        return out

    cand, ctrl = sel["candidate"], sel["control"]
    primary = boot(cand, ctrl)
    l_int = float(np.mean([rows[f"{n}|iiid_lam{lam[ctrl]:g}"]["small"]["macro_logloss"] for n in family_seeds(ctrl, seeds)]))
    p99 = {int(k): v for k, v in manifest["p99_ms_by_family_seed"][cand].items()} if cand in manifest["p99_ms_by_family_seed"] else None
    verdict = D.verdict(primary, l_int, p99)
    result = {"mode": mode, "evaluation_git": git, "code_fingerprint": code_fingerprint(), "files": [str(f) for f in files],
              "files_sha256": verified or {f.name: sha256(f) for f in files}, "freeze_sha256": sha256(run / "freeze.json"),
              "rows": df.height, "rows_by_file": [int((file_of == k).sum()) for k in range(len(files))],
              "rows_by_class": {str(k): int(v) for k, v in zip(*np.unique(cls, return_counts=True))},
              "primary": primary, "l_int": l_int, "verdict": verdict,
              "leave_one_domain_out": D.leave_one_domain_out(y, calibrated(cand), calibrated(ctrl), dom, cls),
              "per_file": {Path(files[k]).name: boot(cand, ctrl, mask=file_of == k) for k in range(len(files))}
              if len(files) > 1 else {},
              "metrics": rows}
    bn, bg = manifest["best_on_v2"]["nn"], manifest["best_on_v2"]["gbdt"]
    result["secondary"] = {"N-star - N-shared": boot("N-star", "N-shared"), "N-head - N-shared": boot("N-head", "N-shared"),
                           "N-star-noPN - N-star": boot("N-star-noPN", "N-star"), f"{bn} - {bg}": boot(bn, bg),
                           f"{cand}: iiid - ii": boot(cand, cand, "iiid", "ii"),
                           f"{ctrl}: iiid - ii": boot(ctrl, ctrl, "iiid", "ii")}
    if cand != "G-shared":
        result["secondary"][f"{cand} - G-shared"] = boot(cand, "G-shared")
    result["table"] = summary_table(rows, lam, seeds, manifest["p99_ms_by_family_seed"])

    if mode == "dryrun":
        result |= dryrun_checks(run, manifest, preds, rows, result)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, default=str))
    (out_dir / "results.md").write_text(render(result, manifest))
    return result


def summary_table(rows: dict, lam: dict, seeds: list[int], p99: dict) -> list[dict]:
    """Small / large loss and raw-row p99 side by side (what would be taken to a serving system)."""
    out = []
    for fam in FAMILIES:
        keys = [f"{n}|iiid_lam{lam[fam]:g}" for n in family_seeds(fam, seeds)]
        out.append({"family": fam,
                    "small_macro_logloss": float(np.mean([rows[k]["small"]["macro_logloss"] for k in keys])),
                    "small_click_logloss": float(np.mean([rows[k]["small"]["click_weighted_logloss"] for k in keys])),
                    "small_poisson_deviance": float(np.mean([rows[k]["small"]["poisson_deviance_mean"] for k in keys])),
                    "large_click_logloss": float(np.mean([rows[k]["large"]["click_weighted_logloss"] for k in keys])),
                    "small_within_domain_auc": float(np.mean([rows[k]["small"]["within_domain_auc_macro"] for k in keys])),
                    "p99_ms_max_over_seeds": max(p99[fam].values()) if fam in p99 else None})
    return out


def dryrun_checks(run: Path, manifest: dict, preds: dict, rows: dict, result: dict) -> dict:
    """The test code must reproduce the development run on V2: predictions, metrics, the primary bootstrap with
    every seed pair matched BY SEED, the verdict derived from them, the secondary bootstraps and the leave-one-out.
    Pooled values alone are not enough: flipping all three seed differences would leave them unchanged."""
    res = json.loads((run / "results.json").read_text())
    checked = 0

    def close(name: str, a: float, b: float) -> None:
        nonlocal checked
        require_close(name, abs(a - b), DRYRUN_TOL)
        checked += 1

    for name, p in preds.items():
        saved = run / "preds" / f"{name}_v2.npy"
        if not saved.exists():
            saved = next((run / "nn_final" / "preds").glob(f"{name}_e*_v2.npy"))
        close(f"prediction {name}", float(np.abs(np.load(saved) - p).max()), 0.0)
    for key, r in rows.items():
        rec = res["v2_rows"].get(key)
        if rec is None:
            raise CheckFailed(f"dry run row {key} is not in the development results")
        for path in COMPARED_METRICS:
            a, b = r, rec
            for k in path:
                a, b = a[k], b[k]
            close(f"{key} {'.'.join(path)}", a, b)

    def same_boot(name: str, new: dict, rec: dict) -> None:
        for k in ("small_macro", "small_click", "large_click", "small_auc_macro"):
            close(f"{name} {k}", new["point"][k], rec["point"][k])
            for a, b in zip(new["ci95"][k], rec["ci95"][k]):
                close(f"{name} CI {k}", a, b)
        ns = dict(zip(new["point"]["per_pair_seeds"], new["point"]["per_pair_small_macro"]))
        rs = dict(zip(rec["point"]["per_pair_seeds"], rec["point"]["per_pair_small_macro"]))
        if sorted(ns) != sorted(rs):
            raise CheckFailed(f"{name}: seed pairs {sorted(ns)} vs recorded {sorted(rs)}")
        for seed in ns:
            close(f"{name} seed {seed}", ns[seed], rs[seed])

    same_boot("primary", result["primary"], res["v2_primary_bootstrap"])
    sel, seeds = manifest["selection"], manifest["seeds"]
    ctrl, lam = sel["control"], sel["lambda_iiid"]
    rec_l_int = float(np.mean([res["v2_rows"][f"{n}|iiid_lam{lam[ctrl]:g}"]["small"]["macro_logloss"]
                               for n in family_seeds(ctrl, seeds)]))
    p99 = {int(k): v for k, v in manifest["p99_ms_by_family_seed"].get(sel["candidate"], {}).items()} or None
    rec_v = D.verdict(res["v2_primary_bootstrap"], rec_l_int, p99)
    for k in ("tier", "improvement_confirmed", "failed_conditions", "pending_conditions"):
        if result["verdict"][k] != rec_v[k]:
            raise CheckFailed(f"verdict {k}: {result['verdict'][k]} vs recorded {rec_v[k]}")
    close("verdict relative improvement", result["verdict"]["conditions"]["effect_size"]["relative_improvement"],
          rec_v["conditions"]["effect_size"]["relative_improvement"])
    bn, bg = manifest["best_on_v2"]["nn"], manifest["best_on_v2"]["gbdt"]
    mapped = secondary_names(result["secondary"], bn, bg)
    if sorted(mapped) != sorted(res["v2_secondary"]):  # a missing comparison must fail too
        raise CheckFailed(f"secondary comparisons {sorted(mapped)} vs recorded {sorted(res['v2_secondary'])}")
    for k, rk in mapped.items():
        same_boot(f"secondary {k}", result["secondary"][rk], res["v2_secondary"][k])
    checked += same_loo(result["leave_one_domain_out"], res["v2_leave_one_domain_out"])
    return {"dryrun_passed": True, "dryrun_values_reproduced": checked}


def secondary_names(secondary: dict, best_nn: str, best_gbdt: str) -> dict[str, str]:
    """Development-run name -> evaluation name (the runner calls the best-of-family comparison 'best NN - best GBDT')."""
    return {("best NN - best GBDT" if k == f"{best_nn} - {best_gbdt}" else k): k for k in secondary}


def same_loo(new: dict, rec: dict) -> int:
    """Leave-one-domain-out reproduced domain by domain, so an emptied per-domain table cannot pass."""
    a = {int(k): v for k, v in new["by_removed_domain"].items()}
    b = {int(k): v for k, v in rec["by_removed_domain"].items()}
    if sorted(a) != sorted(b) or not a:
        raise CheckFailed(f"leave-one-out domains {sorted(a)} vs recorded {sorted(b)}")
    n = 0
    for d in a:
        require_close(f"leave-one-out domain {d}", abs(a[d] - b[d]), DRYRUN_TOL)
        n += 1
    for k in ("full", "min", "max"):
        require_close(f"leave-one-out {k}", abs(new[k] - rec[k]), DRYRUN_TOL)
        n += 1
    if new["sign_flips"] != rec["sign_flips"]:
        raise CheckFailed("leave-one-out sign flips differ")
    return n


def render(r: dict, manifest: dict) -> str:
    p, ci, v = r["primary"]["point"], r["primary"]["ci95"], r["verdict"]
    lines = [f"# Study C final evaluation ({r['mode']}): {', '.join(Path(f).name for f in r['files'])}", "",
             f"- evaluation code `{r['evaluation_git']['commit'][:7]}`" + (" (dirty)" if r["evaluation_git"]["dirty_files"] else ""),
             f"- freeze.json `{r['freeze_sha256'][:16]}`; files {r['files_sha256']}; rows {r['rows']} {r['rows_by_file']}",
             f"- primary: {manifest['primary']['comparison']} on {manifest['primary']['metric']}",
             f"- **difference {p['small_macro']:+.6f}, 95% CI [{ci['small_macro'][0]:+.6f}, {ci['small_macro'][1]:+.6f}], "
             f"per seed pair {[round(x, 6) for x in p['per_pair_small_macro']]} -> tier: {v['tier']}**",
             f"- conditions: {json.dumps(v['conditions'], default=str)}",
             f"- large domains (click-weighted) {p['large_click']:+.6f} [{ci['large_click'][0]:+.6f}, {ci['large_click'][1]:+.6f}]; "
             f"small within-domain AUC {p['small_auc_macro']:+.5f} [{ci['small_auc_macro'][0]:+.5f}, {ci['small_auc_macro'][1]:+.5f}] "
             f"({p['auc_domains']} domains)",
             f"- leave one small domain out: range [{r['leave_one_domain_out']['min']:+.6f}, {r['leave_one_domain_out']['max']:+.6f}], "
             f"sign flips {r['leave_one_domain_out']['sign_flips']}"]
    for f, b in r["per_file"].items():
        lines.append(f"- {f}: {b['point']['small_macro']:+.6f} [{b['ci95']['small_macro'][0]:+.6f}, {b['ci95']['small_macro'][1]:+.6f}]")
    lines += ["", "| family | small macro LogLoss | small click LogLoss | small Poisson deviance (group totals) | "
                  "large click LogLoss | small within-domain AUC | p99 ms, max over seeds (uniform traffic) |",
              "|---|---|---|---|---|---|---|"]
    for t in r["table"]:
        p99 = t["p99_ms_max_over_seeds"]
        lines.append(f"| {t['family']} | {t['small_macro_logloss']:.6f} | {t['small_click_logloss']:.6f} | "
                     f"{t['small_poisson_deviance']:.4f} | {t['large_click_logloss']:.6f} | "
                     f"{t['small_within_domain_auc']:.4f} | {'–' if p99 is None else round(p99, 3)} |")
    lines += ["", "| secondary (A - B) | small macro [95% CI] | large click [95% CI] |", "|---|---|---|"]
    for k, b in r["secondary"].items():
        pp, cc = b["point"], b["ci95"]
        lines.append(f"| {k} | {pp['small_macro']:+.6f} [{cc['small_macro'][0]:+.6f}, {cc['small_macro'][1]:+.6f}] | "
                     f"{pp['large_click']:+.6f} [{cc['large_click'][0]:+.6f}, {cc['large_click'][1]:+.6f}] |")
    lines += ["", "Intervals hold the fitted models fixed (evaluation-user uncertainty only); users seen on both files "
                  "are one cluster."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    cmd, run_arg = sys.argv[1], Path(sys.argv[2])
    if cmd == "expect":
        print(json.dumps(expect(run_arg), indent=2))
    elif cmd == "freeze":
        m = freeze(run_arg)
        print(f"froze {run_arg}: {len(m['files_sha256'])} files; candidate {m['selection']['candidate']}")
    elif cmd == "dryrun":
        cfg_ = json.loads((run_arg / "freeze.json").read_text())["config"]
        d = cfg_["data"]
        v2 = [Path(d["root"]) / (d["smoke_file"] if d["mode"] == "smoke" else d["v2"][0])]
        out = evaluate(run_arg, v2, "dryrun")
        print(json.dumps({"dryrun_passed": out["dryrun_passed"], "values": out["dryrun_values_reproduced"],
                          "verdict": out["verdict"]["tier"]}, indent=2))
    elif cmd == "test":
        out = evaluate(run_arg, [Path(a) for a in sys.argv[3:]], "test")
        print(json.dumps({"primary": out["primary"]["point"], "ci95": out["primary"]["ci95"],
                          "verdict": out["verdict"]}, indent=2, default=str))
    else:
        raise SystemExit(__doc__)
