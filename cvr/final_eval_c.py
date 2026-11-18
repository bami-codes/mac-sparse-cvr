"""Study C: freeze a development run, dry-run the evaluation on V2, then evaluate train-13 + train-14 ONCE.

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
test    verifies every frozen hash, requires a passed dry run and a clean working tree, refuses to run twice,
        accepts only the frozen holdout files, evaluates both files together (a user seen on both days is one
        bootstrap cluster) plus each file's direction, and selects or tunes nothing.

Networks are scored in a PyTorch-only subprocess (cvr/final_eval_c_nn.py); this process loads LightGBM.
"""

import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import polars as pl

from . import domain_eval as D
from . import groups as G
from .artifacts import load_gbdt
from .checks import CheckFailed, require_close, require_finite_probs
from .domain_models import load_domain_calibrators, load_gbdt_offset
from .features import featurize, read_raw
from .mddata import DOM_COL, encode_md, load_md, save_md
from .schema import CAT_INPUTS, DOMAIN_KEY, GROUP_KEY

DRYRUN_TOL = 1e-9
FAMILIES = ["G-int", "G-shared", "N-int", "N-shared", "N-head", "N-star", "N-star-noPN"]
HOLDOUT = ["train-13.parquet", "train-14.parquet"]
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
                                "config.toml")]
    return {str(p.relative_to(run)): sha256(p) for p in files}


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
    lat = json.loads((run / "latency_c.json").read_text())["families"]
    p99 = {}
    for fam in ("G-int", "G-shared", "N-int", "N-shared", "N-head", "N-star"):
        f = lat[fam]
        if sorted(int(s) for s in f["seeds"]) != sorted(D.SEEDS):
            raise CheckFailed(f"latency of {fam} measured for seeds {sorted(f['seeds'])}")
        p99[fam] = {int(s): float(v) for s, v in f["p99_ms_by_seed"].items()}
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


def evaluate(run: Path, files: list[Path], mode: str) -> dict:
    manifest = verify_freeze(run)
    cfg, sel, seeds = manifest["config"], manifest["selection"], manifest["seeds"]
    out_dir = run / "final" / mode
    git = git_state()
    if mode == "test":
        if out_dir.exists():
            raise CheckFailed(f"{out_dir} exists: the holdout files are evaluated only once")
        if git["dirty_files"]:
            raise CheckFailed(f"uncommitted changes; commit the evaluation code before the test: {git['dirty_files'][:5]}")
        dry = run / "final" / "dryrun" / "results.json"
        if not dry.exists() or not json.loads(dry.read_text()).get("dryrun_passed"):
            raise CheckFailed("the V2 dry run must pass before the test")
        if sorted(f.name for f in files) != sorted(manifest["holdout_files"]):
            raise CheckFailed(f"test files {[f.name for f in files]} differ from the frozen holdout {manifest['holdout_files']}")
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
    result = {"mode": mode, "evaluation_git": git, "files": [str(f) for f in files],
              "files_sha256": {f.name: sha256(f) for f in files}, "freeze_sha256": sha256(run / "freeze.json"),
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
        result |= dryrun_checks(run, preds, rows, primary)
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
                    "large_click_logloss": float(np.mean([rows[k]["large"]["click_weighted_logloss"] for k in keys])),
                    "small_within_domain_auc": float(np.mean([rows[k]["small"]["within_domain_auc_macro"] for k in keys])),
                    "p99_ms_max_over_seeds": max(p99[fam].values()) if fam in p99 else None})
    return out


def dryrun_checks(run: Path, preds: dict, rows: dict, primary: dict) -> dict:
    """The test code must reproduce the development run on V2: predictions, metrics and the primary bootstrap."""
    res = json.loads((run / "results.json").read_text())
    checked = 0
    for name, p in preds.items():
        saved = run / "preds" / f"{name}_v2.npy"
        if not saved.exists():
            saved = next((run / "nn_final" / "preds").glob(f"{name}_e*_v2.npy"))
        require_close(f"dryrun prediction {name}", float(np.abs(np.load(saved) - p).max()), DRYRUN_TOL)
        checked += 1
    for key, r in rows.items():
        rec = res["v2_rows"].get(key)
        if rec is None:
            raise CheckFailed(f"dry run row {key} is not in the development results")
        for path in COMPARED_METRICS:
            a, b = r, rec
            for k in path:
                a, b = a[k], b[k]
            require_close(f"dryrun {key} {'.'.join(path)}", abs(a - b), DRYRUN_TOL)
            checked += 1
    rec = res["v2_primary_bootstrap"]
    for k in ("small_macro", "small_click", "large_click", "small_auc_macro"):
        require_close(f"dryrun primary {k}", abs(primary["point"][k] - rec["point"][k]), DRYRUN_TOL)
        require_close(f"dryrun primary CI {k}", max(abs(a - b) for a, b in zip(primary["ci95"][k], rec["ci95"][k])),
                      DRYRUN_TOL)
        checked += 3
    return {"dryrun_passed": True, "dryrun_values_reproduced": checked}


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
    lines += ["", "| family | small macro LogLoss | large click LogLoss | small within-domain AUC | p99 ms (max over seeds) |",
              "|---|---|---|---|---|"]
    for t in r["table"]:
        lines.append(f"| {t['family']} | {t['small_macro_logloss']:.6f} | {t['large_click_logloss']:.6f} | "
                     f"{t['small_within_domain_auc']:.4f} | {t['p99_ms_max_over_seeds'] if t['p99_ms_max_over_seeds'] is None else round(t['p99_ms_max_over_seeds'], 3)} |")
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
    if cmd == "freeze":
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
