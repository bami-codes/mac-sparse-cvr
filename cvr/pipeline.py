"""One command: vocabulary -> features -> groups -> models -> calibration -> V2 selection -> results.

    uv run python -m cvr.pipeline configs/smoke_day17.toml

The test file is never read here. Selection (base model, lambda, calibration method) is made
on V2 only and frozen in selection.json before any test evaluation.
"""

import json
import platform
import resource
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import numpy as np
import polars as pl
import scipy
import sklearn

from . import groups as G
from . import metrics as M
from . import vocab as V
from .calibration import FreqAware, GlobalPlatt, GroupShrunkPlatt
from .features import featurize, read_raw
from .models import GBDT, LogReg, SmoothedRate
from .schema import CAT_INPUTS, GROUP_KEY, SHOP_KEY, VOCAB_COLS

ROLES = ["train", "v1", "v2"]


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty_files": run("status", "--porcelain").splitlines()}


def sources(cfg: dict) -> dict[str, list[tuple[Path, tuple[int, int] | None]]]:
    d = cfg["data"]
    root = Path(d["root"])
    if d["mode"] == "smoke":  # one file, users split into disjoint buckets
        f = root / d["smoke_file"]
        return {r: [(f, tuple(d[f"smoke_{r}_buckets"]))] for r in ROLES}
    if "test" in d:
        raise ValueError("the test file must not be listed in a development config")
    return {r: [(root / f, None) for f in d[r]] for r in ROLES}


def sha_of(path: Path) -> str | None:
    side = path.with_name(path.name + ".sha256")
    return side.read_text().split()[0] if side.exists() else None


def evaluate(df: pl.DataFrame, p: np.ndarray) -> dict:
    y = df["y"].to_numpy()
    g = df[GROUP_KEY].to_numpy()
    cls = df["group_class"].to_numpy()
    nonmiss = df["is_missing"].to_numpy() == 0
    g1 = cls == "G1"
    return {
        "G1": M.group_report(y[g1], p[g1], g[g1]),
        "overall_nonmissing": M.overall(y[nonmiss], p[nonmiss]),
        "overall_all": M.overall(y, p),
        "pooled_O_over_E_by_class": {c: float(y[cls == c].sum() / p[cls == c].sum())
                                     for c in G.CLASSES if (cls == c).any()},
    }


def mean_sd(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0, "n": int(a.size)}


def main(config_path: str) -> Path:
    t0 = time.time()
    cfg = tomllib.loads(Path(config_path).read_text())
    run_dir = Path("runs") / cfg["name"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True)
    (run_dir / "config.toml").write_text(Path(config_path).read_text())
    timings: dict[str, float] = {}
    srcs = sources(cfg)
    user_pct = cfg["data"].get("smoke_user_pct")

    # 1) one vocabulary from the training sources only
    counts_frames = [read_raw(p, b, user_pct, columns=VOCAB_COLS).select(VOCAB_COLS) for p, b in srcs["train"]]
    vocab = V.build_vocab(V.count_values(counts_frames), min_count=cfg["vocab"]["min_count"])
    del counts_frames
    V.save(vocab, run_dir / "vocab")
    sizes = V.sizes(vocab)
    timings["vocab"] = time.time() - t0

    # 2) compact features per role, all encoded with the same vocabulary
    data: dict[str, pl.DataFrame] = {}
    for role in ROLES:
        data[role] = pl.concat([featurize(read_raw(p, b, user_pct), vocab) for p, b in srcs[role]])
    timings["features"] = time.time() - t0 - sum(timings.values())

    # 3) groups from training statistics only
    gc = cfg["groups"]
    stats = G.train_stats(data["train"])
    for role in ROLES:
        data[role] = pl.concat([data[role], G.classify(data[role], stats, gc["n_min"], gc["c_min"], gc["c_dense"])],
                               how="horizontal")
    group_summary = G.summary(stats, gc["n_min"], gc["c_min"], gc["c_dense"])
    group_summary["rows_by_class"] = {r: data[r]["group_class"].value_counts().sort("group_class").to_dicts()
                                      for r in ROLES}

    # 4) base models
    mc = cfg["models"]
    preds: dict[str, dict[str, np.ndarray]] = {r: {} for r in ("v1", "v2")}
    fitted_info: dict[str, dict] = {}
    for name, key in (("M0a", GROUP_KEY), ("M0b", SHOP_KEY)):
        m = SmoothedRate.fit(data["train"], key)
        fitted_info[name] = {"key": key, "p0": m.p0, "prior_strength_m": m.m}
        for r in preds:
            preds[r][name] = m.predict(data[r])
    t1 = time.time()
    lr = LogReg.fit(data["train"], sizes, mc["lr_C"], mc["lr_max_iter"])
    fitted_info["M1"] = {"C": mc["lr_C"], "n_iter": int(lr.model.n_iter_[0]), "converged": lr.converged,
                         "dims": lr.n_onehot, "seconds": time.time() - t1}
    for r in preds:
        preds[r]["M1"] = lr.predict(data[r])
    variants = {"M2": None}
    if mc.get("ablation_no_history"):
        variants["M2_nohist"] = CAT_INPUTS  # same model without purchase-history inputs
    for vname, inputs in variants.items():
        for seed in mc["seeds"]:
            t1 = time.time()
            gb = GBDT.fit(data["train"], mc["lgb"], seed, mc["num_boost_round"], mc["early_stopping_rounds"], inputs)
            fitted_info[f"{vname}_s{seed}"] = {"best_iteration": gb.booster.best_iteration, "seconds": time.time() - t1}
            for r in preds:
                preds[r][f"{vname}_s{seed}"] = gb.predict(data[r])
    timings["models"] = time.time() - t0 - sum(timings.values())

    # 5) calibration on V1, evaluation on V2
    v1, v2 = data["v1"], data["v2"]
    y1 = v1["y"].to_numpy()
    g1_codes, g2_codes = v1[GROUP_KEY].to_numpy(), v2[GROUP_KEY].to_numpy()
    n1, n2 = v1["tr_clicks"].to_numpy(), v2["tr_clicks"].to_numpy()
    lambdas = cfg["calibration"]["lambdas"]
    base_names = ["M1"] + [f"M2_s{s}" for s in mc["seeds"]]
    rows: dict[str, dict] = {}
    for name in ["M0a", "M0b"] + list(preds["v2"]):
        rows[f"{name}|raw"] = evaluate(v2, preds["v2"][name])
    cal_info: dict[str, dict] = {}
    for name in base_names:
        p1, p2 = preds["v1"][name], preds["v2"][name]
        glob = GlobalPlatt.fit(p1, y1)
        rows[f"{name}|ii"] = evaluate(v2, glob.predict(p2))
        for lam in lambdas:
            gs = GroupShrunkPlatt.fit(p1, y1, g1_codes, glob, lam)
            rows[f"{name}|iii_lam{lam:g}"] = evaluate(v2, gs.predict(p2, g2_codes))
            cal_info[f"{name}|iii_lam{lam:g}"] = {"nonpositive_slopes": gs.n_nonpositive_slopes,
                                                  "converged": gs.converged, "groups": int(gs.group_ids.size)}
        fa = FreqAware.fit(p1, y1, n1)
        rows[f"{name}|iv"] = evaluate(v2, fa.predict(p2, n2))
        cal_info[name] = {"ii": {"a": glob.a, "b": glob.b}, "iv": {"w": fa.w.tolist(), "w0": fa.w0}}
    timings["calibration"] = time.time() - t0 - sum(timings.values())

    # 6) selection on V2 only (seed-averaged primary metric), frozen before any test run
    def primary(key: str) -> float:
        return rows[key]["G1"]["macro_logloss"]

    def seed_mean(family: str, suffix: str) -> float:
        names = ["M1"] if family == "M1" else [f"M2_s{s}" for s in mc["seeds"]]
        return float(np.mean([primary(f"{n}|{suffix}") for n in names]))

    base = min(["M1", "M2"], key=lambda f: seed_mean(f, "raw"))
    lam = min(lambdas, key=lambda l: seed_mean(base, f"iii_lam{l:g}"))
    method = min([f"iii_lam{lam:g}", "iv"], key=lambda s: seed_mean(base, s))
    selection = {
        "selected_on": "V2 primary metric (G1 macro LogLoss), averaged over base-model seeds",
        "base_model": base, "lambda_iii": lam, "calibration_method": method,
        "seed_means": {f"{f}|{s}": seed_mean(f, s) for f in ("M1", "M2")
                       for s in ["raw", "ii", "iv"] + [f"iii_lam{l:g}" for l in lambdas]},
        "groups": group_summary["definition"], "vocab_fingerprint": V.fingerprint(vocab),
    }

    # informational paired bootstrap on V2 (selection data, NOT a confirmatory test)
    bs = cfg["bootstrap"]
    first = "M1" if base == "M1" else f"M2_s{mc['seeds'][0]}"
    glob = GlobalPlatt.fit(preds["v1"][first], y1)
    p_ii = glob.predict(preds["v2"][first])
    p_sel = (FreqAware.fit(preds["v1"][first], y1, n1).predict(preds["v2"][first], n2) if method == "iv" else
             GroupShrunkPlatt.fit(preds["v1"][first], y1, g1_codes, glob, lam).predict(preds["v2"][first], g2_codes))
    g1m = v2["group_class"].to_numpy() == "G1"
    boot = M.paired_bootstrap_macro_diff(v2["y"].to_numpy()[g1m], p_sel[g1m], p_ii[g1m], g2_codes[g1m],
                                         v2["user_key"].to_numpy()[g1m], bs["reps"], bs["seed"])
    boot["note"] = f"{first}: selected ({method}) minus (ii) on V2 G1 rows; selection data, not a test"
    timings["selection_bootstrap"] = time.time() - t0 - sum(timings.values())

    meta = {
        "config": config_path, "git": git_state(), "created_utc": run_dir.name,
        "data": {r: [{"file": str(p), "buckets": b, "sha256": sha_of(p)} for p, b in srcs[r]] for r in ROLES},
        "smoke_user_pct": user_pct, "rows": {r: data[r].height for r in ROLES},
        "vocab": {"fingerprint": V.fingerprint(vocab), "sizes": sizes, "min_count": cfg["vocab"]["min_count"]},
        "versions": {"python": platform.python_version(), "polars": pl.__version__, "lightgbm": lightgbm.__version__,
                     "sklearn": sklearn.__version__, "scipy": scipy.__version__, "numpy": np.__version__},
        "timings_seconds": {k: round(v, 1) for k, v in timings.items()} | {"total": round(time.time() - t0, 1)},
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
        "fitted": fitted_info, "calibration": cal_info,
    }
    out = {"meta": meta, "groups": group_summary, "selection": selection, "v2_bootstrap_info": boot, "v2_rows": rows}
    (run_dir / "results.json").write_text(json.dumps(out, indent=2, default=str))
    (run_dir / "selection.json").write_text(json.dumps(selection, indent=2, default=str))
    (run_dir / "results.md").write_text(render_markdown(out, mc["seeds"], lambdas))
    print(f"wrote {run_dir}")
    return run_dir


def render_markdown(out: dict, seeds: list[int], lambdas: list[float]) -> str:
    meta, gs, sel, rows = out["meta"], out["groups"], out["selection"], out["v2_rows"]
    d = gs["definition"]
    lines = [
        f"# Results: {Path(meta['config']).stem} (V2 = selection data, not a test)", "",
        f"- commit `{meta['git']['commit'][:7]}`" + (" (dirty working tree)" if meta["git"]["dirty_files"] else ""),
        f"- rows: {meta['rows']}, vocab `{meta['vocab']['fingerprint']}`, smoke user % = {meta['smoke_user_pct']}",
        f"- data: " + "; ".join(f"{r}: " + ", ".join(Path(x['file']).name + (f" buckets {x['buckets']}" if x['buckets'] else "")
                                                   for x in meta['data'][r]) for r in ROLES),
        f"- **G1 groups**: `{d['key']}` with training clicks ≥ {d['n_min']} and training positives < {d['c_min']} "
        f"(selected on training files only): {gs['G1_groups']} groups, {gs['G1_train_click_share']:.2%} of training clicks",
        f"- selection: base **{sel['base_model']}**, λ(iii) = {sel['lambda_iii']:g}, method **{sel['calibration_method']}**",
        f"- time {meta['timings_seconds']['total']} s, peak memory {meta['peak_rss_gb']} GB", "",
        "| model | calib | G1 macro LogLoss | G1 pooled O/E | G1 median abs(O/E−1) | G1 within-group AUC | LogLoss (non-missing) | AUC (non-missing) | O/E (non-missing) | ECE10 | AUC (all rows) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    families = {"M0a": ["M0a"], "M0b": ["M0b"], "M1": ["M1"], "M2": [f"M2_s{s}" for s in seeds],
                "M2_nohist": [f"M2_nohist_s{s}" for s in seeds]}
    suffixes = ["raw", "ii"] + [f"iii_lam{l:g}" for l in lambdas] + ["iv"]
    for fam, names in families.items():
        for suf in suffixes:
            keys = [f"{n}|{suf}" for n in names if f"{n}|{suf}" in rows]
            if not keys:
                continue

            def cell(path: tuple[str, str], fmt: str = ".5f") -> str:
                vals = [rows[k][path[0]][path[1]] for k in keys]
                if any(v is None for v in vals):
                    return "–"
                s = mean_sd(vals)
                return f"{s['mean']:{fmt}}" + (f" ± {s['sd']:{fmt}}" if s["n"] > 1 else "")

            lines.append(f"| {fam} | {suf} | {cell(('G1', 'macro_logloss'))} | {cell(('G1', 'pooled_O_over_E'), '.3f')} | "
                         f"{cell(('G1', 'median_abs_ratio_dev'), '.3f')} | {cell(('G1', 'within_group_auc_macro'), '.4f')} | "
                         f"{cell(('overall_nonmissing', 'logloss'))} | {cell(('overall_nonmissing', 'auc'), '.4f')} | "
                         f"{cell(('overall_nonmissing', 'O_over_E'), '.3f')} | {cell(('overall_nonmissing', 'ece10'), '.5f')} | "
                         f"{cell(('overall_all', 'auc'), '.4f')} |")
    b = out["v2_bootstrap_info"]
    lines += ["", f"Paired user bootstrap on V2 G1 rows ({b['note']}): diff = {b['diff']:+.6f}, "
                  f"95% CI [{b['ci95'][0]:+.6f}, {b['ci95'][1]:+.6f}], {b['reps']} reps, {b['users']} users, {b['groups']} groups.",
              "", "Values with ± are mean ± sd over base-model seeds."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main(sys.argv[1])
