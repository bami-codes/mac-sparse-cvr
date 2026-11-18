"""Freeze a development run, dry-run the evaluation on V2, then evaluate the test file ONCE.

    uv run python -m cvr.final_eval freeze  runs/<name>/<stamp>
    uv run python -m cvr.final_eval dryrun  runs/<name>/<stamp>                 # V2 of the run
    uv run python -m cvr.final_eval test    runs/<name>/<stamp> data/raw/mac/test/test.parquet

freeze   recomputes the training group statistics with the saved vocabulary (and checks them against
         the saved calibrator tables), then writes freeze.json: SHA-256 of every model / calibrator /
         vocabulary file, the G1 list, the selection, the primary comparison and success criteria,
         and the secondary comparisons. freeze.json is never overwritten.
dryrun   runs the exact evaluation code on the run's V2 and requires every recorded V2 metric of the
         pipeline to be reproduced (hard check), so the code is verified end to end before the test.
test     verifies every frozen hash, refuses to run twice, evaluates the pre-defined comparisons in
         one pass, and never selects or tunes anything.
"""

import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import polars as pl

from . import groups as G
from . import metrics as M
from .artifacts import load_gbdt, load_logreg, load_stacking
from .calibration import load_calibrators
from .checks import CheckFailed, require_close, require_commit, require_finite_probs
from .features import featurize, read_raw
from .pipeline import evaluate, git_state, mean_sd, sources
from .schema import GROUP_KEY, MISSING

DRYRUN_TOL = 1e-9


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def vocab_frames(folder: Path) -> dict[str, pl.DataFrame]:
    return {p.stem: pl.read_parquet(p) for p in folder.glob("*.parquet")}


def family_names(base: str, seeds: list[int]) -> list[str]:
    return ["M1"] if base == "M1" else [f"{base}_s{s}" for s in seeds]


def hashed_files(run: Path) -> dict[str, str]:
    files = [p for d in ("models", "vocab", "id_vocab") if (run / d).exists() for p in sorted((run / d).rglob("*"))
             if p.is_file()]
    files.append(run / "frozen_group_stats.parquet")
    return {str(p.relative_to(run)): sha256(p) for p in files}


# ------------------------------------------------------------------ freeze
def freeze(run: Path) -> dict:
    out = run / "freeze.json"
    if out.exists():
        raise CheckFailed(f"{out} already exists; a frozen run is never re-frozen")
    cfg = tomllib.loads((run / "config.toml").read_text())
    res = json.loads((run / "results.json").read_text())
    sel = json.loads((run / "selection.json").read_text())
    vocab = vocab_frames(run / "vocab")
    user_pct = cfg["data"].get("smoke_user_pct")
    train = pl.concat([featurize(read_raw(p, b, user_pct), vocab).select(GROUP_KEY, "y")
                       for p, b in sources(cfg)["train"]])
    stats, p0 = G.train_stats(train), float(train["y"].mean())
    # the recomputed statistics must match the tables saved with the calibrators
    saved = pl.read_parquet(next((run / "models").glob("calib_*.group_clicks.parquet")))
    cmp = stats.join(saved.rename({"tr_clicks": "saved"}), on=GROUP_KEY, how="full", coalesce=True)
    if cmp.filter(pl.col("tr_clicks") != pl.col("saved")).height or cmp.null_count().sum_horizontal().item():
        raise CheckFailed("recomputed training group statistics differ from the saved calibrator tables")
    stats.sort(GROUP_KEY).write_parquet(run / "frozen_group_stats.parquet")
    gc, seeds = cfg["groups"], cfg["models"]["seeds"]
    g1 = stats.filter((pl.col(GROUP_KEY) != MISSING) & (pl.col("tr_clicks") >= gc["n_min"]) & (pl.col("tr_pos") < gc["c_min"]))
    base, method = sel["base_model"], sel["calibration_method"]
    m3 = (cfg["models"].get("m3") or {}).get("enabled", False)
    manifest = {
        "run": str(run), "pipeline_git_commit": res["meta"]["git"]["commit"],
        "pipeline_dirty_files": res["meta"]["git"]["dirty_files"],
        "config": cfg, "seeds": seeds,
        "selection": {"base_model": base, "calibration_method": method, "lambda_iii": sel["lambda_iii"],
                      "selected_on": sel["selected_on"], "mode": sel.get("selection_mode", "calibration"),
                      "primary": sel.get("primary") or {
                          "type": "calibration", "description": f"{base}: {method} vs (ii)",
                          "treatment": [f"{n}|{method}" for n in family_names(base, seeds)],
                          "control": [f"{n}|ii" for n in family_names(base, seeds)]}},
        "groups": {"definition": gc, "p0_train": p0, "G1_groups": g1.height,
                   "G1_codes_sha256": hashlib.sha256(np.sort(g1[GROUP_KEY].to_numpy()).tobytes()).hexdigest()},
        "primary": {
            "comparison": (sel.get("primary") or {}).get("description")
                          or f"{base} family ({len(family_names(base, seeds))} seed(s)): {method} vs (ii) global Platt",
            "metric": "G1 macro LogLoss (equal weight over G1 groups), lower is better",
            "uncertainty": f"seed-paired user bootstrap, {cfg['bootstrap']['reps']} reps, seed {cfg['bootstrap']['seed']}; "
                           "models fixed (evaluation-user uncertainty only)",
            "success": "selected minus (ii) < 0, 95% CI upper bound < 0, and every seed pair < 0",
        },
        "secondary_criteria": {
            "overall_auc_drop": "seed-mean non-missing AUC of selected minus (ii) >= -0.002 (project provisional)",
            "group_calibration": "G1 median |O/E - 1| of selected <= (ii)",
            "latency": "batch=1 p99 <= 1 ms, single thread, raw row -> calibrated probability (latency.json)",
        },
        "secondary_comparisons": [
            "history ablation M1 vs M1_nohist and M2 vs M2_nohist (raw, seed-paired bootstrap)",
        ] + (["M3 attention vs mean pooling (raw, seed-paired bootstrap)"] if m3 else []) + [
            "model families raw / (ii) / selected / (iv): G1 macro LogLoss, AUC, GAUC, O/E",
            "G1exp sensitivity definition (expected positives) for the same comparisons",
        ],
        "rule": "the test file is evaluated once; nothing is selected or tuned on it",
    }
    manifest["files_sha256"] = hashed_files(run)
    out.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def verify_freeze(run: Path) -> dict:
    manifest = json.loads((run / "freeze.json").read_text())
    now = hashed_files(run)
    bad = sorted(k for k in set(now) | set(manifest["files_sha256"])
                 if now.get(k) != manifest["files_sha256"].get(k))
    if bad:
        raise CheckFailed(f"frozen files changed since freeze: {bad[:10]}")
    return manifest


# ------------------------------------------------------------------ evaluation
def evaluate_file(run: Path, eval_file: Path, mode: str) -> dict:
    manifest = verify_freeze(run)
    cfg, seeds = manifest["config"], manifest["seeds"]
    out_dir = run / "final" / mode
    git = git_state()
    if mode == "test" and out_dir.exists():
        raise CheckFailed(f"{out_dir} exists: the test file is evaluated only once")
    if mode == "test" and git["dirty_files"]:
        raise CheckFailed(f"uncommitted changes; commit the evaluation code before the test: {git['dirty_files'][:5]}")
    if mode == "test":
        require_commit(git, "the test")
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke = cfg["data"]["mode"] == "smoke"
    buckets = tuple(cfg["data"]["smoke_v2_buckets"]) if smoke and mode == "dryrun" else None
    user_pct = cfg["data"].get("smoke_user_pct") if smoke and mode == "dryrun" else None

    vocab = vocab_frames(run / "vocab")
    df = featurize(read_raw(eval_file, buckets, user_pct), vocab)
    stats = pl.read_parquet(run / "frozen_group_stats.parquet")
    gc = manifest["groups"]["definition"]
    df = pl.concat([df, G.classify(df, stats, gc["n_min"], gc["c_min"], gc["c_dense"], manifest["groups"]["p0_train"])],
                   how="horizontal")

    mdir = run / "models"
    if (mdir / "stacking").exists():  # second study: out-of-fold inputs are built from full-training fits
        df = load_stacking(mdir / "stacking").transform(df)
    preds: dict[str, np.ndarray] = {}
    for spec in sorted(mdir.glob("M*.json")):
        name = spec.stem
        if name.startswith("M1"):
            preds[name] = load_logreg(mdir / name).predict(df)
        elif name.startswith("M2"):
            preds[name] = load_gbdt(mdir / name).predict(df)
    if any(mdir.glob("M3_*.json")):  # PyTorch-only process
        args = [sys.executable, "-m", "cvr.final_eval_m3", str(run), str(eval_file), str(out_dir / "m3_preds")]
        if buckets:
            args += [str(buckets[0]), str(buckets[1]), str(user_pct)]
        subprocess.run(args, check=True)
        for f in sorted((out_dir / "m3_preds").glob("*.npy")):
            preds[f.stem] = np.load(f)

    sel = manifest["selection"]
    lam_key = f"iii_lam{sel['lambda_iii']:g}"
    g = df[GROUP_KEY].to_numpy()
    n = df.select(pl.col("tr_clicks")).to_series().to_numpy()
    g1m = df["group_class"].to_numpy() == "G1"
    rows, g1_pred = {}, {}

    def record(key: str, p: np.ndarray) -> None:
        require_finite_probs(key, p)  # every model and every calibrator separately, on all rows
        rows[key] = evaluate(df, p)
        g1_pred[key] = p[g1m]

    for name, p in preds.items():
        record(f"{name}|raw", p)
        calib = mdir / f"calib_{name}.npz"
        if calib.exists():
            glob_, gs, fa, _ = load_calibrators(calib.with_suffix(""))
            record(f"{name}|ii", glob_.predict(p))
            record(f"{name}|{lam_key}", gs.predict(p, g))
            record(f"{name}|iv", fa.predict(p, n))

    if mode == "dryrun":  # the evaluation code must reproduce the pipeline's own V2 numbers
        recorded = json.loads((run / "results.json").read_text())["v2_rows"]
        checked = 0
        for key, r in rows.items():
            if key in recorded:
                for part in ("G1", "overall_nonmissing", "overall_all"):
                    for metric in ("macro_logloss", "logloss", "auc", "O_over_E", "pooled_O_over_E", "gauc"):
                        a, b = r[part].get(metric), recorded[key][part].get(metric)
                        if a is not None and b is not None:
                            require_close(f"dryrun {key} {part}.{metric}", abs(a - b), DRYRUN_TOL)
                            checked += 1
        if checked == 0:
            raise CheckFailed("dry-run compared nothing")
        dryrun_checked = checked

    # ---------------- primary comparison (pre-defined in freeze.json)
    bs = cfg["bootstrap"]
    y1, u1, g1 = df["y"].to_numpy()[g1m], df["user_key"].to_numpy()[g1m], g[g1m]
    pc = sel.get("primary") or {  # runs frozen before the primary definition was stored (first study)
        "treatment": [f"{n_}|{sel['calibration_method']}" for n_ in family_names(sel["base_model"], seeds)],
        "control": [f"{n_}|ii" for n_ in family_names(sel["base_model"], seeds)]}
    prim = M.paired_bootstrap_macro_diff(y1, [g1_pred[k] for k in pc["treatment"]],
                                         [g1_pred[k] for k in pc["control"]], g1, u1, bs["reps"], bs["seed"])
    if mode == "dryrun":  # same rows, order and bootstrap seed -> must equal the pipeline's V2 bootstrap
        rec = json.loads((run / "results.json").read_text())["v2_bootstrap_info"]
        require_close("dryrun primary diff", abs(prim["diff"] - rec["diff"]), DRYRUN_TOL)
        require_close("dryrun primary CI", max(abs(a - b) for a, b in zip(prim["ci95"], rec["ci95"])), DRYRUN_TOL)
    auc_drop = mean_sd([rows[t]["overall_nonmissing"]["auc"] - rows[c]["overall_nonmissing"]["auc"]
                        for t, c in zip(pc["treatment"], pc["control"])])["mean"]
    cal_sel = mean_sd([rows[t]["G1"]["median_abs_ratio_dev"] for t in pc["treatment"]])["mean"]
    cal_ii = mean_sd([rows[c]["G1"]["median_abs_ratio_dev"] for c in pc["control"]])["mean"]
    bid_t = mean_sd([rows[t]["G1"]["abs_conv_error_share"] for t in pc["treatment"]])["mean"]
    bid_c = mean_sd([rows[c]["G1"]["abs_conv_error_share"] for c in pc["control"]])["mean"]
    primary = prim | {"success": bool(prim["diff"] < 0 and prim["ci95"][1] < 0 and all(d < 0 for d in prim["per_pair_diff"])),
                      "overall_auc_selected_minus_ii": auc_drop, "auc_criterion_met": bool(auc_drop >= -0.002),
                      "G1_median_abs_ratio_dev": {"selected": cal_sel, "ii": cal_ii},
                      "group_calibration_criterion_met": bool(cal_sel <= cal_ii),
                      "G1_aggregate_error_share": {"treatment": bid_t, "control": bid_c}}

    # ---------------- secondary comparisons (pre-defined)
    secondary = {}
    if "M1_nohist" in preds:
        secondary["history_M1"] = M.paired_bootstrap_macro_diff(y1, g1_pred["M1|raw"], g1_pred["M1_nohist|raw"], g1, u1,
                                                                bs["reps"], bs["seed"])
    if all(f"M2_nohist_s{s}" in preds for s in seeds):
        secondary["history_M2"] = M.paired_bootstrap_macro_diff(
            y1, [g1_pred[f"M2_s{s}|raw"] for s in seeds], [g1_pred[f"M2_nohist_s{s}|raw"] for s in seeds],
            g1, u1, bs["reps"], bs["seed"])
    if all(f"M3_{v}_s{s}" in preds for v in ("attn", "mean") for s in seeds):
        secondary["m3_attn_vs_mean"] = M.paired_bootstrap_macro_diff(
            y1, [g1_pred[f"M3_attn_s{s}|raw"] for s in seeds], [g1_pred[f"M3_mean_s{s}|raw"] for s in seeds],
            g1, u1, bs["reps"], bs["seed"])

    result = {"mode": mode, "dryrun_metrics_reproduced": dryrun_checked if mode == "dryrun" else None,
              "evaluation_git": git, "eval_file": str(eval_file), "eval_file_sha256": sha256(eval_file),
              "freeze_sha256": sha256(run / "freeze.json"), "rows": df.height,
              "rows_by_class": df["group_class"].value_counts().sort("group_class").to_dicts(),
              "primary": primary, "secondary": secondary, "metrics": rows}
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, default=str))
    (out_dir / "results.md").write_text(render(result, manifest))
    return result


def render(r: dict, manifest: dict) -> str:
    seeds, sel = manifest["seeds"], manifest["selection"]
    lam_key = f"iii_lam{sel['lambda_iii']:g}"
    p = r["primary"]
    lines = [f"# Final evaluation ({r['mode']}): {Path(r['eval_file']).name}", "",
             f"- evaluation code `{r['evaluation_git']['commit'][:7]}`" + (" (dirty)" if r['evaluation_git']['dirty_files'] else ""),
             f"- freeze.json sha256 `{r['freeze_sha256'][:16]}`, eval file sha256 `{r['eval_file_sha256'][:16]}`, rows {r['rows']}",
             f"- primary: {manifest['primary']['comparison']} on {manifest['primary']['metric']}",
             f"- **diff = {p['diff']:+.6f}, 95% CI [{p['ci95'][0]:+.6f}, {p['ci95'][1]:+.6f}], per seed pair "
             f"{[round(x, 6) for x in p['per_pair_diff']]} → success: {p['success']}**",
             f"- overall AUC (selected − ii) {p['overall_auc_selected_minus_ii']:+.5f} (criterion met: {p['auc_criterion_met']}); "
             f"G1 median |O/E−1| treatment {p['G1_median_abs_ratio_dev']['selected']:.4f} vs control {p['G1_median_abs_ratio_dev']['ii']:.4f}; "
             f"G1 group aggregate error sum|E-O|/sum O treatment {p['G1_aggregate_error_share']['treatment']:.4f} vs control {p['G1_aggregate_error_share']['control']:.4f}", "",
             "| family | calib | G1 macro LogLoss | G1exp macro LogLoss | AUC (non-missing) | GAUC | O/E (non-missing) |",
             "|---|---|---|---|---|---|---|"]
    fams = {"M1": ["M1"], "M1_nohist": ["M1_nohist"], "M2": [f"M2_s{s}" for s in seeds],
            "M2_nohist": [f"M2_nohist_s{s}" for s in seeds], "M2aux": [f"M2aux_s{s}" for s in seeds],
            "M2te": [f"M2te_s{s}" for s in seeds], "M2both": [f"M2both_s{s}" for s in seeds],
            "M3_attn": [f"M3_attn_s{s}" for s in seeds],
            "M3_mean": [f"M3_mean_s{s}" for s in seeds]}
    for fam, names in fams.items():
        for suf in ("raw", "ii", lam_key, "iv"):
            keys = [f"{n}|{suf}" for n in names if f"{n}|{suf}" in r["metrics"]]
            if not keys:
                continue

            def c(part, metric, fmt=".5f"):
                s = mean_sd([r["metrics"][k][part][metric] for k in keys])
                return f"{s['mean']:{fmt}}" + (f" ± {s['sd']:{fmt}}" if s["n"] > 1 else "")

            lines.append(f"| {fam} | {suf} | {c('G1', 'macro_logloss')} | {c('G1exp', 'macro_logloss')} | "
                         f"{c('overall_nonmissing', 'auc', '.4f')} | {c('overall_nonmissing', 'gauc', '.4f')} | "
                         f"{c('overall_nonmissing', 'O_over_E', '.3f')} |")
    lines.append("")
    for k, v in r["secondary"].items():
        lines.append(f"- {k}: diff {v['diff']:+.6f}, 95% CI [{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}], "
                     f"per pair {[round(x, 6) for x in v['per_pair_diff']]}")
    lines += ["", "Intervals hold the fitted models fixed (evaluation-user uncertainty only)."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    cmd, run_arg = sys.argv[1], Path(sys.argv[2])
    if cmd == "freeze":
        m = freeze(run_arg)
        print(f"froze {run_arg}: {len(m['files_sha256'])} files, G1 {m['groups']['G1_groups']} groups")
    elif cmd == "dryrun":
        cfg_ = json.loads((run_arg / "freeze.json").read_text())["config"]
        v2 = Path(cfg_["data"]["root"]) / (cfg_["data"]["smoke_file"] if cfg_["data"]["mode"] == "smoke"
                                            else cfg_["data"]["v2"][0])
        print(json.dumps(evaluate_file(run_arg, v2, "dryrun")["primary"], indent=2))
    elif cmd == "test":
        print(json.dumps(evaluate_file(run_arg, Path(sys.argv[3]), "test")["primary"], indent=2))
    else:
        raise SystemExit(__doc__)
