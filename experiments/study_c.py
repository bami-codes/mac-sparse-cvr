"""Study C development run (docs/DESIGN.md §16): vocabulary -> features -> domain classes -> GBDT family (G-int,
G-shared) -> NN tuning job -> NN final job -> (ii) / (iii-d) on V1 -> V2 selection -> power check -> results.

    uv run python -m experiments.study_c experiments/configs/c_dev_train08_10.toml

The holdout files are never read here: domain_eval.check_development_sources refuses a config that lists
train-13 / train-14 / test, uses one file in two roles, or overlaps smoke buckets. V2 chooses the NN
learning rate / epochs, every lambda and the single interpretation candidate; selection.json freezes the choice
before any test file is opened. PyTorch runs only in the worker processes (experiments/study_c_worker.py).
"""

import json
import platform
import subprocess
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import lightgbm
import numpy as np
import polars as pl

from cvr import metrics as M
from cvr.calibration import GlobalPlatt, fit_group_shrunk_exact, load_group_calibrators, save_group_calibrators
from cvr.checks import RELOAD_TOL, require_close, require_finite_probs
from cvr.schema import CAT_INPUTS, DOMAIN_KEY, GROUP_KEY
from cvr.sysinfo import peak_rss_gb

from . import domain_eval as D
from . import groups as G
from . import vocab as V
from .artifacts import load_gbdt, save_gbdt
from .domain_models import G_SHARED_INPUTS, GBDTOffset, load_gbdt_offset, save_gbdt_offset
from .features import featurize, read_raw
from .mddata import DOM_COL, concat_md, encode_md, save_md
from .models import GBDT
from .pipeline import ROLES, git_state, sha_of, sources

NN_FAMILIES = {"N-int": "int", "N-shared": "shared", "N-head": "head", "N-star": "star"}
ABLATION = ("N-star-noPN", "star_nopn")  # secondary analysis only, trained with N-star's chosen settings
RELOAD_ROWS = 50_000


def family_seeds(fam: str, seeds: list[int]) -> list[str]:
    return [f"{fam}_s{s}" for s in seeds]


def run_worker(job_dir: Path, job: dict) -> dict:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.json").write_text(json.dumps(job, indent=2))
    subprocess.run([sys.executable, "-m", "experiments.study_c_worker", str(job_dir)], check=True)
    return json.loads((job_dir / "result.json").read_text())


def main(config_path: str) -> Path:
    t0 = time.time()
    git_at_start = git_state()
    cfg = tomllib.loads(Path(config_path).read_text())
    run_dir = Path("runs") / cfg["name"] / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True)
    (run_dir / "config.toml").write_text(Path(config_path).read_text())
    timings: dict[str, float] = {}

    def lap(name: str) -> None:
        timings[name] = round(time.time() - t0 - sum(timings.values()), 1)

    D.check_development_sources(cfg["data"])
    srcs = sources(cfg)
    user_pct = cfg["data"].get("smoke_user_pct")
    mc, nc, dc = cfg["models"], cfg["models"]["nn"], cfg["domains"]
    seeds = mc["seeds"]

    # 1) vocabulary from the training sources only; 2) features and network arrays with that vocabulary
    counts = None
    for p, b in srcs["train"]:
        frame = read_raw(p, b, user_pct, columns=V.VOCAB_COLS).select(V.VOCAB_COLS)
        counts = V.count_values([frame], counts=counts)
        del frame
    vocab = V.build_vocab(counts, min_count=cfg["vocab"]["min_count"])
    del counts
    V.save(vocab, run_dir / "vocab")
    sizes = V.sizes(vocab)
    data: dict[str, pl.DataFrame] = {}
    nn_dir = run_dir / "nn_data"
    for role in ROLES:
        parts = [featurize(read_raw(p, b, user_pct), vocab) for p, b in srcs[role]]
        data[role] = pl.concat(parts) if len(parts) > 1 else parts[0]
        save_md(concat_md([encode_md(f) for f in parts]) if len(parts) > 1 else encode_md(parts[0]), nn_dir / role)
        del parts
    np.save(nn_dir / "y_train.npy", data["train"]["y"].to_numpy().astype(np.float32))
    lap("vocab_features")

    # 3) domain classes (and G1 for the secondary metric) from training statistics only
    dstats = D.domain_stats(data["train"])
    cls = {r: D.domain_class(data[r], dstats, dc["small_share"], dc["small_min_clicks"]) for r in ROLES}
    dom = {r: data[r][DOMAIN_KEY].to_numpy() for r in ROLES}
    domain_summary = D.domain_summary(dstats, dc["small_share"], dc["small_min_clicks"])
    domain_summary["rows_by_class"] = {
        r: {str(k): int(v) for k, v in zip(*np.unique(cls[r], return_counts=True))} for r in ROLES
    }
    gc = cfg["groups"]
    gstats = G.train_stats(data["train"])
    g1 = {
        r: G.classify(data[r], gstats, gc["n_min"], gc["c_min"], gc["c_dense"])["group_class"].to_numpy() == "G1"
        for r in ("v1", "v2")
    }
    lap("groups")

    # 4) GBDT family (this process): every seed saved and reloaded on the V2 head
    preds: dict[str, dict[str, np.ndarray]] = {"v1": {}, "v2": {}}
    fitted: dict[str, dict] = {}
    reload_check: dict[str, float] = {}
    models_dir = run_dir / "models"
    models_dir.mkdir()
    preds_dir = run_dir / "preds"
    preds_dir.mkdir()
    head = data["v2"].head(RELOAD_ROWS)
    lgb_params = mc["lgb"]
    for seed in seeds:
        for fam, fit, save, load in (
            (
                "G-shared",
                lambda s: GBDT.fit(
                    data["train"], lgb_params, s, mc["num_boost_round"], mc["early_stopping_rounds"], G_SHARED_INPUTS
                ),
                save_gbdt,
                load_gbdt,
            ),
            (
                "G-int",
                lambda s: GBDTOffset.fit(
                    data["train"], lgb_params, s, mc["num_boost_round"], mc["early_stopping_rounds"]
                ),
                save_gbdt_offset,
                load_gbdt_offset,
            ),
        ):
            t1 = time.time()
            m = fit(seed)
            name = f"{fam}_s{seed}"
            fitted[name] = {"rounds_from_early_stopping": m.rounds, "inputs": m.inputs, "seconds": time.time() - t1}
            for r in preds:
                preds[r][name] = m.predict(data[r])
                require_finite_probs(f"{name} {r}", preds[r][name])
                np.save(preds_dir / f"{name}_{r}.npy", preds[r][name])  # so selection can be recomputed later
            save(m, models_dir / name)
            p_re = load(models_dir / name).predict(head)
            reload_check[name] = require_close(
                f"reload {name}", float(np.abs(p_re - preds["v2"][name][: head.height]).max()), RELOAD_TOL
            )
            print(f"{name}: {m.rounds} rounds, {time.time() - t1:.0f}s", flush=True)
    lap("gbdt")

    # 5) NN learning rate / epochs on V2 (seed 1, every compared variant, same grid and metric), then all seeds
    job_base = {
        "cat_sizes": [sizes[c] for c in CAT_INPUTS],
        "dom_col": DOM_COL,
        "dim": nc["dim"],
        "batch": nc["batch"],
        "threads": nc["threads"],
        "reload_rows": RELOAD_ROWS,
        "data_dir": str(nn_dir),
        "models_dir": str(models_dir),
    }
    tune_runs = [
        {
            "name": f"{fam}_lr{lr:g}_s{seeds[0]}",
            "variant": var,
            "seed": seeds[0],
            "lr": lr,
            "epochs": max(nc["epochs"]),
            "eval_epochs": nc["epochs"],
            "save": False,
        }
        for fam, var in NN_FAMILIES.items()
        for lr in nc["lrs"]
    ]
    run_worker(run_dir / "nn_tune", job_base | {"runs": tune_runs})  # predictions are read from nn_tune/preds
    y1, y2 = data["v1"]["y"].to_numpy(), data["v2"]["y"].to_numpy()
    nonmiss2 = data["v2"]["is_missing"].to_numpy() == 0
    tuning: dict[str, dict] = {}
    for fam in NN_FAMILIES:
        grid = {}
        for lr in nc["lrs"]:
            for e in nc["epochs"]:
                key = f"{fam}_lr{lr:g}_s{seeds[0]}_e{e}"
                p1 = np.load(run_dir / "nn_tune" / "preds" / f"{key}_v1.npy")
                p2 = np.load(run_dir / "nn_tune" / "preds" / f"{key}_v2.npy")
                q2 = GlobalPlatt.fit(p1, y1).predict(p2)
                grid[f"lr{lr:g}_e{e}"] = {
                    "lr": lr,
                    "epochs": e,
                    "v2_logloss_nonmissing_ii": float(M.bce(y2[nonmiss2], q2[nonmiss2]).mean()),
                }
        best = min(grid, key=lambda k: grid[k]["v2_logloss_nonmissing_ii"])  # grid order breaks exact ties
        tuning[fam] = {"grid": grid, "chosen": grid[best], "rule": "min V2 non-missing LogLoss after (ii), seed 1"}
    final_runs = []
    for fam, var in list(NN_FAMILIES.items()) + [ABLATION]:
        ch = tuning["N-star" if fam == ABLATION[0] else fam]["chosen"]
        final_runs += [
            {
                "name": f"{fam}_s{s}",
                "variant": var,
                "seed": s,
                "lr": ch["lr"],
                "epochs": ch["epochs"],
                "eval_epochs": [ch["epochs"]],
                "save": True,
            }
            for s in seeds
        ]
    final = run_worker(run_dir / "nn_final", job_base | {"runs": final_runs})
    nn_names: dict[str, str] = {}  # family_seed -> saved model name (with the epoch suffix)
    for run in final_runs:
        saved = f"{run['name']}_e{run['epochs']}"
        nn_names[run["name"]] = saved
        fitted[run["name"]] = final["fitted"][saved]
        reload_check[run["name"]] = require_close(f"reload {saved}", final["reload_check"][saved], RELOAD_TOL)
        for r in preds:
            preds[r][run["name"]] = np.load(run_dir / "nn_final" / "preds" / f"{saved}_{r}.npy")
    # the final seed-1 fit must reproduce the tuning checkpoint it was chosen from (determinism check)
    repro = {}
    for fam in NN_FAMILIES:
        ch = tuning[fam]["chosen"]
        tuned = np.load(run_dir / "nn_tune" / "preds" / f"{fam}_lr{ch['lr']:g}_s{seeds[0]}_e{ch['epochs']}_v2.npy")
        repro[fam] = require_close(
            f"{fam} seed-1 final vs tuning checkpoint",
            float(np.abs(preds["v2"][f"{fam}_s{seeds[0]}"] - tuned).max()),
            RELOAD_TOL,
        )
    lap("nn")

    # 6) calibration on V1 ((ii) and (iii-d) with every lambda), reports on V2
    families = ["G-int", "G-shared"] + list(NN_FAMILIES) + [ABLATION[0]]
    lambdas = cfg["calibration"]["lambdas"]
    v2 = data["v2"]
    users2 = v2["user_key"].to_numpy()
    g1_groups2 = v2[GROUP_KEY].to_numpy()
    cal: dict[str, dict] = {}
    cal_solver: dict[str, dict] = {}
    rows: dict[str, dict] = {}

    def rep(key: str, p: np.ndarray) -> None:  # metrics only: calibrated V2 predictions are recomputed when needed
        rows[key] = D.report(y2, p, dom["v2"], cls["v2"], nonmiss2, g1["v2"], g1_groups2)

    def v2_pred(name: str, calib: str, lam: float | None = None) -> np.ndarray:
        p2 = preds["v2"][name]
        if calib == "raw":
            return p2
        if calib == "ii":
            return cal[name]["ii"].predict(p2)
        return cal[name]["iiid"][lam].predict(p2, dom["v2"])

    for fam in families:
        for name in family_seeds(fam, seeds):
            p1, p2 = preds["v1"][name], preds["v2"][name]
            glob = GlobalPlatt.fit(p1, y1)
            cal[name] = {"ii": glob, "iiid": {}}
            rep(f"{name}|raw", p2)
            rep(f"{name}|ii", glob.predict(p2))
            for lam in lambdas:
                gs, sinfo = fit_group_shrunk_exact(p1, y1, dom["v1"], glob, lam)  # exact per-domain solve
                if gs.n_nonpositive_slopes or not gs.converged:
                    raise RuntimeError(
                        f"(iii-d) fit of {name}, lambda {lam}: converged={gs.converged}, "
                        f"non-positive slopes={gs.n_nonpositive_slopes}, {sinfo}"
                    )
                cal[name]["iiid"][lam] = gs
                cal_solver[f"{name}|lam{lam:g}"] = sinfo
                rep(f"{name}|iiid_lam{lam:g}", gs.predict(p2, dom["v2"]))
    lap("calibration")

    # 7) V2 selection (seed means of the primary metric), primary comparison, power check, verdict dry run
    def seed_mean(fam: str, suffix: str, path=("small", "macro_logloss")) -> float:
        return float(np.mean([rows[f"{n}|{suffix}"][path[0]][path[1]] for n in family_seeds(fam, seeds)]))

    lam_of = {fam: min(lambdas, key=lambda x: seed_mean(fam, f"iiid_lam{x:g}")) for fam in families}
    scores = {fam: seed_mean(fam, f"iiid_lam{lam_of[fam]:g}") for fam in D.PREFERENCE}
    cand = D.choose_candidate(scores)
    ctrl = D.INT_OF[cand]

    def calibrated(fam: str, calib: str = "iiid") -> list[np.ndarray]:
        return [v2_pred(n, calib, lam_of[fam]) for n in family_seeds(fam, seeds)]

    bs = cfg["bootstrap"]

    def boot(a: str, b: str, ca: str = "iiid", cb: str = "iiid") -> dict:
        out = D.paired_bootstrap(
            y2,
            calibrated(a, ca),
            calibrated(b, cb),
            dom["v2"],
            cls["v2"],
            users2,
            bs["reps"],
            bs["seed"],
            pair_seeds=seeds,
        )
        out["compared"] = f"{a} ({ca}) - {b} ({cb})"
        return out

    primary = boot(cand, ctrl)
    l_int = seed_mean(ctrl, f"iiid_lam{lam_of[ctrl]:g}")
    lo, hi = primary["ci95"]["small_macro"]
    power = D.power_check((hi - lo) / 2, l_int)
    verdict_dry = D.verdict(primary, l_int, p99_by_seed=None) | {
        "note": "DRY RUN of the verdict code on V2 = selection data; not a result (latency measured later)"
    }
    loo = D.leave_one_domain_out(y2, calibrated(cand), calibrated(ctrl), dom["v2"], cls["v2"])
    best_g = min(["G-int", "G-shared"], key=lambda f: seed_mean(f, f"iiid_lam{lam_of[f]:g}"))
    best_n = min(list(NN_FAMILIES), key=lambda f: seed_mean(f, f"iiid_lam{lam_of[f]:g}"))
    secondary = {
        "N-star - N-shared": boot("N-star", "N-shared"),
        "N-head - N-shared": boot("N-head", "N-shared"),
        "N-star-noPN - N-star": boot("N-star-noPN", "N-star"),
        "best NN - best GBDT": boot(best_n, best_g),
        f"{cand}: iiid - ii": boot(cand, cand, "iiid", "ii"),
        f"{ctrl}: iiid - ii": boot(ctrl, ctrl, "iiid", "ii"),
    }
    if cand != "G-shared":  # what a serving system would actually get: the candidate vs G-shared
        secondary[f"{cand} - G-shared"] = boot(cand, "G-shared")
    selection = {
        "selected_on": "V2 small-domain macro LogLoss (seed mean) after (iii-d), docs/DESIGN.md §16.6",
        "nn_settings": {fam: tuning[fam]["chosen"] for fam in NN_FAMILIES} | {ABLATION[0]: tuning["N-star"]["chosen"]},
        "lambda_iiid": lam_of,
        "candidate_scores": scores,
        "candidate": cand,
        "control": ctrl,
        "tie_rule": {"tolerance": D.TIE, "preference": D.PREFERENCE},
        "primary": {
            "treatment": [f"{n}|iiid_lam{lam_of[cand]:g}" for n in family_seeds(cand, seeds)],
            "control": [f"{n}|iiid_lam{lam_of[ctrl]:g}" for n in family_seeds(ctrl, seeds)],
            "description": f"{cand} (iii-d) - {ctrl} (iii-d), small-domain macro LogLoss",
        },
        "domains": domain_summary["definition"] | {k: domain_summary[k]["codes"] for k in ("small", "tiny", "large")},
        "vocab_fingerprint": V.fingerprint(vocab),
        "lgb_params": lgb_params,
        "nn": nc,
        "seeds": seeds,
        "adoption": {
            "relative": D.ADOPT_REL,
            "large_ci_upper": D.ADOPT_LARGE,
            "p99_ms": D.ADOPT_P99_MS,
            "p99_rule": "max over the selected candidate's seeds",
        },
        "power_check": power,
    }

    # 8) calibrators of every base model at its family's chosen lambda, saved and reloaded
    for fam in families:
        for name in family_seeds(fam, seeds):
            glob, gs = cal[name]["ii"], cal[name]["iiid"][lam_of[fam]]
            save_group_calibrators(glob, gs, models_dir / f"calib_{name}")
            g2, s2 = load_group_calibrators(models_dir / f"calib_{name}")
            ph, dh = preds["v2"][name][:RELOAD_ROWS], dom["v2"][:RELOAD_ROWS]
            for tag, a, b in (
                ("ii", g2.predict(ph), glob.predict(ph)),
                ("iiid", s2.predict(ph, dh), gs.predict(ph, dh)),
            ):
                require_finite_probs(f"reloaded calib_{name}|{tag}", a)
                reload_check[f"calib_{name}|{tag}"] = require_close(
                    f"reload calib_{name}|{tag}", float(np.abs(a - b).max()), RELOAD_TOL
                )
    lap("selection")

    meta = {
        "config": config_path,
        "git": git_at_start,
        "created_utc": run_dir.name,
        "data": {r: [{"file": str(p), "buckets": b, "sha256": sha_of(p)} for p, b in srcs[r]] for r in ROLES},
        "smoke_user_pct": user_pct,
        "rows": {r: data[r].height for r in ROLES},
        "vocab": {"fingerprint": V.fingerprint(vocab), "sizes": sizes},
        "versions": {
            "python": platform.python_version(),
            "polars": pl.__version__,
            "lightgbm": lightgbm.__version__,
            "numpy": np.__version__,
        },
        "timings_seconds": timings | {"total": round(time.time() - t0, 1)},
        "peak_rss_gb": peak_rss_gb(),
        "fitted": fitted,
        "reload_check_max_abs_diff": reload_check,
        "nn_seed1_reproduction": repro,
        "calibration_solver": {
            "solver": "per-group Newton (cvr/calibration.fit_group_shrunk_exact)",
            "max_projected_gradient": max(v["max_projected_gradient"] for v in cal_solver.values()),
            "max_iterations": max(v["max_iterations"] for v in cal_solver.values()),
            "fits_with_a_group_at_the_slope_bound": sum(v["groups_at_slope_bound"] > 0 for v in cal_solver.values()),
        },
    }
    out = {
        "meta": meta,
        "domains": domain_summary,
        "nn_tuning": tuning,
        "selection": selection,
        "v2_primary_bootstrap": primary,
        "v2_verdict_dry_run": verdict_dry,
        "v2_leave_one_domain_out": loo,
        "v2_secondary": secondary,
        "v2_rows": rows,
    }
    (run_dir / "results.json").write_text(json.dumps(out, indent=2, default=str))
    (run_dir / "selection.json").write_text(json.dumps(selection, indent=2, default=str))
    (run_dir / "results.md").write_text(render_markdown(out, families, seeds, lambdas))
    print(f"wrote {run_dir}")
    return run_dir


def render_markdown(out: dict, families: list[str], seeds: list[int], lambdas: list[float]) -> str:
    meta, ds, sel, rows = out["meta"], out["domains"], out["selection"], out["v2_rows"]
    d = ds["definition"]
    lines = [
        f"# Study C development run: {Path(meta['config']).stem} (V2 = selection data, not a test)",
        "",
        f"- commit `{meta['git']['commit'][:7]}`" + (" (dirty working tree)" if meta["git"]["dirty_files"] else ""),
        f"- rows {meta['rows']}, vocab `{meta['vocab']['fingerprint']}`, smoke user % {meta['smoke_user_pct']}",
        f"- domains `{d['key']}` (training rows only): small = share < {d['small_share']:.0%} and clicks ≥ "
        f"{d['small_min_clicks']}: **{ds['small']['domains']}** ({ds['small']['train_click_share']:.2%} of training "
        f"clicks, {ds['small']['train_positives']} positives); tiny {ds['tiny']['domains']}; large {ds['large']['domains']}",
        f"- time {meta['timings_seconds']['total']} s ({meta['timings_seconds']}), peak memory {meta['peak_rss_gb']} GB",
        "",
        "## NN learning rate / epochs (seed 1, V2 non-missing LogLoss after (ii))",
        "",
        "| family | " + " | ".join(out["nn_tuning"]["N-int"]["grid"]) + " | chosen |",
        "|---|" + "---|" * (len(out["nn_tuning"]["N-int"]["grid"]) + 1),
    ]
    for fam, t in out["nn_tuning"].items():
        lines.append(
            f"| {fam} | "
            + " | ".join(f"{v['v2_logloss_nonmissing_ii']:.6f}" for v in t["grid"].values())
            + f" | lr {t['chosen']['lr']:g}, {t['chosen']['epochs']} ep |"
        )
    lines += [
        "",
        "## V2 table (seed mean ± sd)",
        "",
        "| family | calib | small macro LogLoss | small click LogLoss | small Poisson dev | small within-domain AUC | "
        "large click LogLoss | LogLoss (non-missing) | AUC (non-missing) | G1 macro LogLoss |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def cell(keys: list[str], path: tuple, fmt: str = ".6f") -> str:
        vals = []
        for k in keys:
            v = rows[k]
            for p in path:
                v = v.get(p) if isinstance(v, dict) else None
            vals.append(v)
        if any(v is None for v in vals):
            return "–"
        a = np.asarray(vals, float)
        return f"{a.mean():{fmt}}" + (f" ± {a.std(ddof=1):{fmt}}" if a.size > 1 else "")

    for fam in families:
        lam = sel["lambda_iiid"][fam]
        for suf in ["raw", "ii", f"iiid_lam{lam:g}"]:
            keys = [f"{n}|{suf}" for n in family_seeds(fam, seeds)]
            lines.append(
                f"| {fam} | {suf} | {cell(keys, ('small', 'macro_logloss'))} | "
                f"{cell(keys, ('small', 'click_weighted_logloss'))} | {cell(keys, ('small', 'poisson_deviance_mean'), '.4f')} | "
                f"{cell(keys, ('small', 'within_domain_auc_macro'), '.4f')} | {cell(keys, ('large', 'click_weighted_logloss'))} | "
                f"{cell(keys, ('overall_nonmissing', 'logloss'))} | {cell(keys, ('overall_nonmissing', 'auc'), '.4f')} | "
                f"{cell(keys, ('G1_macro_logloss',))} |"
            )
    pb, pw, vd = out["v2_primary_bootstrap"], sel["power_check"], out["v2_verdict_dry_run"]
    pt, ci = pb["point"], pb["ci95"]
    lines += [
        "",
        "## Selection (V2)",
        "",
        f"- lambda (iii-d) per family: {sel['lambda_iiid']}",
        f"- candidate scores (small macro LogLoss after iii-d): { {k: round(v, 6) for k, v in sel['candidate_scores'].items()} }",
        f"- **candidate {sel['candidate']}**, control **{sel['control']}**",
        f"- primary on V2 ({pb['compared']}): small macro {pt['small_macro']:+.6f} [{ci['small_macro'][0]:+.6f}, "
        f"{ci['small_macro'][1]:+.6f}], per seed pair {[round(x, 6) for x in pt['per_pair_small_macro']]}; "
        f"large click {pt['large_click']:+.6f} [{ci['large_click'][0]:+.6f}, {ci['large_click'][1]:+.6f}]; "
        f"small within-domain AUC {pt.get('small_auc_macro', float('nan')):+.5f} "
        f"[{ci.get('small_auc_macro', [np.nan, np.nan])[0]:+.5f}, {ci.get('small_auc_macro', [np.nan, np.nan])[1]:+.5f}] "
        f"({pt.get('auc_domains')} domains with both classes)",
        f"- power check: h_V2 {pw['h_v2']:.6f}, optimistic h_test {pw['h_test_optimistic']:.6f}, detectable at 80% "
        f"{pw['detectable_80pct']:.6f} vs adoption effect {pw['adoption_effect_abs']:.6f} -> "
        f"underpowered expected: {pw['underpowered_expected']}",
        f"- verdict code DRY RUN on V2 (not a result): tier {vd['tier']}, failed {vd['failed_conditions']}, "
        f"pending {vd['pending_conditions']}",
        f"- leave one small domain out: full {out['v2_leave_one_domain_out']['full']:+.6f}, range "
        f"[{out['v2_leave_one_domain_out']['min']:+.6f}, {out['v2_leave_one_domain_out']['max']:+.6f}], sign flips "
        f"{out['v2_leave_one_domain_out']['sign_flips']}",
        "",
        "## Secondary comparisons on V2 (A - B)",
        "",
        "| comparison | small macro [95% CI] | per seed pair | large click [95% CI] |",
        "|---|---|---|---|",
    ]
    for k, b in out["v2_secondary"].items():
        p, c = b["point"], b["ci95"]
        lines.append(
            f"| {k} | {p['small_macro']:+.6f} [{c['small_macro'][0]:+.6f}, {c['small_macro'][1]:+.6f}] | "
            f"{[round(x, 6) for x in p['per_pair_small_macro']]} | {p['large_click']:+.6f} "
            f"[{c['large_click'][0]:+.6f}, {c['large_click'][1]:+.6f}] |"
        )
    lines += [
        "",
        "Intervals resample V2 users with the fitted models held fixed (seed-paired); V2 chose every setting, "
        "so the selected candidate's values are optimistic.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main(sys.argv[1])
