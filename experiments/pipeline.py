"""One command: vocabulary -> features -> groups -> models -> calibration -> V2 selection -> results.

    uv run python -m experiments.pipeline experiments/configs/smoke_day17.toml

The test file is never read here. Selection (base model, lambda, calibration method) is made
on V2 only and frozen in selection.json before any test evaluation.
"""

import json
import platform
import shutil
import subprocess
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import lightgbm
import numpy as np
import polars as pl
import scipy
import sklearn

from cvr import metrics as M
from cvr.calibration import FreqAware, GlobalPlatt, GroupShrunkPlatt, load_calibrators, save_calibrators
from cvr.checks import RELOAD_TOL, require_close, require_finite_probs
from cvr.schema import CAT_INPUTS, GROUP_KEY, NUM_INPUTS, SEQ_PAIRS, SHOP_KEY, VOCAB_COLS
from cvr.sysinfo import peak_rss_gb

from . import artifacts as A
from . import groups as G
from . import vocab as V
from .artifacts import load_stacking, save_stacking
from .features import featurize, read_raw
from .m3data import ID_COLS, OTHER_CATS, concat_arrays, count_ids, encode_m3, save_arrays
from .models import GBDT, LogReg, SmoothedRate
from .stacking import AUX_INPUTS, TE_INPUTS, fit_stacking

ROLES = ["train", "v1", "v2"]
RELOAD_ROWS = 50_000  # V2 rows used to compare in-memory vs reloaded predictions


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
    extra = {}
    if "group_class_exp" in df.columns:  # sensitivity definition (expected positives)
        ge = df["group_class_exp"].to_numpy() == "G1exp"
        extra["G1exp"] = M.group_report(y[ge], p[ge], g[ge])
    return extra | {
        "G1": M.group_report(y[g1], p[g1], g[g1]),
        "overall_nonmissing": M.overall(y[nonmiss], p[nonmiss])
        | M.gauc(y[nonmiss], p[nonmiss], df["user_key"].to_numpy()[nonmiss]),
        "overall_all": M.overall(y, p),
        "pooled_O_over_E_by_class": {
            c: float(y[cls == c].sum() / p[cls == c].sum()) for c in G.CLASSES if (cls == c).any()
        },
    }


def mean_sd(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0, "n": int(a.size)}


def main(config_path: str) -> Path:
    t0 = time.time()
    git_at_start = git_state()  # recorded before any work, so later edits do not leak into provenance
    cfg = tomllib.loads(Path(config_path).read_text())
    run_dir = Path("runs") / cfg["name"] / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True)
    (run_dir / "config.toml").write_text(Path(config_path).read_text())
    timings: dict[str, float] = {}
    srcs = sources(cfg)
    user_pct = cfg["data"].get("smoke_user_pct")

    # 1) one vocabulary from the training sources only, counted one file at a time
    counts = None
    for p, b in srcs["train"]:
        frame = read_raw(p, b, user_pct, columns=VOCAB_COLS).select(VOCAB_COLS)
        counts = V.count_values([frame], counts=counts)
        del frame
    vocab = V.build_vocab(counts, min_count=cfg["vocab"]["min_count"])
    del counts
    V.save(vocab, run_dir / "vocab")
    sizes = V.sizes(vocab)
    # M3 has its own id vocabulary (targets and purchase sequences); M1/M2 codes are untouched
    m3c = cfg["models"].get("m3", {})
    use_m3 = bool(m3c.get("enabled"))
    if use_m3:
        idc = None
        for p, b in srcs["train"]:
            frame = read_raw(p, b, user_pct, columns=[t for t, _ in SEQ_PAIRS] + [s for _, s in SEQ_PAIRS])
            idc = count_ids(frame, idc)
            del frame
        id_vocab = V.build_vocab(idc, min_count=m3c["id_min_count"])
        del idc
        V.save(id_vocab, run_dir / "id_vocab")
    timings["vocab"] = time.time() - t0

    # 2) compact features per role, all encoded with the same vocabulary (+ M3 arrays from the same raw rows)
    data: dict[str, pl.DataFrame] = {}
    arrays: dict = {}
    for role in ROLES:
        parts, aparts = [], []
        for p, b in srcs[role]:
            raw = read_raw(p, b, user_pct)
            f = featurize(raw, vocab)
            parts.append(f)
            if use_m3:
                aparts.append(encode_m3(raw, f, id_vocab))
            del raw
        data[role] = pl.concat(parts)
        if use_m3:
            arrays[role] = concat_arrays(aparts) if len(aparts) > 1 else aparts[0]
    timings["features"] = time.time() - t0 - sum(timings.values())

    # 3) groups from training statistics only
    gc = cfg["groups"]
    stats = G.train_stats(data["train"])
    p0_train = float(data["train"]["y"].mean())  # all training rows, for the expected-positive definition
    for role in ROLES:
        data[role] = pl.concat(
            [data[role], G.classify(data[role], stats, gc["n_min"], gc["c_min"], gc["c_dense"], p0_train)],
            how="horizontal",
        )
    group_summary = G.summary(stats, gc["n_min"], gc["c_min"], gc["c_dense"], p0_train)
    group_summary["rows_by_class"] = {
        r: data[r]["group_class"].value_counts().sort("group_class").to_dicts() for r in ROLES
    }

    # 4) base models (first-seed models are saved and immediately reloaded to check predictions)
    mc = cfg["models"]
    preds: dict[str, dict[str, np.ndarray]] = {r: {} for r in ("v1", "v2")}
    fitted_info: dict[str, dict] = {}
    models_dir = run_dir / "models"
    models_dir.mkdir()
    reload_check: dict[str, float] = {}

    def v2_head() -> pl.DataFrame:
        return data["v2"].head(RELOAD_ROWS)

    def reload_diff(name: str, p_reloaded: np.ndarray, p_memory: np.ndarray) -> float:
        """Max |reloaded - in-memory| on the V2 head; raises CheckFailed above RELOAD_TOL or on NaN."""
        require_finite_probs(f"reloaded {name}", p_reloaded)
        return require_close(
            f"reload {name}", float(np.abs(p_reloaded - p_memory[: p_reloaded.size]).max()), RELOAD_TOL
        )

    for name, key in (("M0a", GROUP_KEY), ("M0b", SHOP_KEY)):
        m = SmoothedRate.fit(data["train"], key)
        fitted_info[name] = {"key": key, "p0": m.p0, "prior_strength_m": m.m}
        for r in preds:
            preds[r][name] = m.predict(data[r])
    ablate = bool(mc.get("ablation_no_history"))
    lr_variants = ({"M1": NUM_INPUTS} | ({"M1_nohist": []} if ablate else {})) if mc.get("include_m1", True) else {}
    for vname, num_inputs in lr_variants.items():
        t1 = time.time()
        lr = LogReg.fit(data["train"], sizes, mc["lr_C"], mc["lr_max_iter"], num_inputs)
        fitted_info[vname] = {
            "C": mc["lr_C"],
            "n_iter": int(lr.model.n_iter_[0]),
            "converged": lr.converged,
            "dims": lr.n_onehot + len(num_inputs),
            "seconds": time.time() - t1,
        }
        for r in preds:
            preds[r][vname] = lr.predict(data[r])
        # every model is saved so the final test can reuse exactly these fits
        A.save_logreg(lr, models_dir / vname)
        reload_check[vname] = reload_diff(
            vname, A.load_logreg(models_dir / vname).predict(v2_head()), preds["v2"][vname]
        )
    # LightGBM: compare the PRE-DEFINED candidates with the first seed on V2 (raw primary metric),
    # then train every seed with the chosen candidate. Candidates override the shared `lgb` params.
    lgb_search = {}
    candidates = {c["name"]: {k: v for k, v in c.items() if k != "name"} for c in mc.get("lgb_candidates", [])}
    if candidates:
        for cname, over in candidates.items():
            t1 = time.time()
            gb = GBDT.fit(
                data["train"],
                mc["lgb"] | over,
                mc["seeds"][0],
                mc["num_boost_round"],
                mc["early_stopping_rounds"],
                CAT_INPUTS + NUM_INPUTS,
            )
            rep = evaluate(data["v2"], gb.predict(data["v2"]))
            lgb_search[cname] = {
                "params": over,
                "rounds": gb.rounds,
                "seconds": time.time() - t1,
                "v2_G1_macro_logloss": rep["G1"]["macro_logloss"],
                "v2_auc_nonmissing": rep["overall_nonmissing"]["auc"],
            }
        chosen = min(lgb_search, key=lambda c: lgb_search[c]["v2_G1_macro_logloss"])
        lgb_params = mc["lgb"] | candidates[chosen]
        lgb_search = {"chosen": chosen, "rule": "min V2 G1 macro LogLoss, first seed, raw", "candidates": lgb_search}
    else:
        lgb_params = mc["lgb"]
    # second study (docs/DESIGN.md §13): out-of-fold auxiliary / group-rate inputs
    stc = mc.get("stacking") or {}
    stacking_info = {}
    if stc.get("enabled"):
        t1 = time.time()
        use_aux, use_te = bool(stc.get("aux", True)), bool(stc.get("te", True))
        data["train"], st_models, stacking_info = fit_stacking(
            data["train"],
            lgb_params,
            stc["aux_seed"],
            stc["folds"],
            mc["num_boost_round"],
            mc["early_stopping_rounds"],
            use_aux,
            use_te,
        )
        for r in ("v1", "v2"):
            data[r] = st_models.transform(data[r])
        save_stacking(st_models, models_dir / "stacking")
        re = load_stacking(models_dir / "stacking").transform(v2_head())
        for c in [c for c in AUX_INPUTS + TE_INPUTS if c in re.columns]:
            stacking_info.setdefault("reload_check", {})[c] = require_close(
                f"reload stacking {c}",
                float(np.abs(re[c].to_numpy() - data["v2"][c].to_numpy()[: re.height]).max()),
                RELOAD_TOL,
            )
        stacking_info["seconds"] = time.time() - t1
    base_inputs = CAT_INPUTS + NUM_INPUTS
    gb_variants = {"M2": base_inputs} | ({"M2_nohist": CAT_INPUTS} if ablate else {})
    if stc.get("enabled"):
        gb_variants |= {
            k: v
            for k, v in {
                "M2aux": base_inputs + AUX_INPUTS,
                "M2te": base_inputs + TE_INPUTS,
                "M2both": base_inputs + AUX_INPUTS + TE_INPUTS,
            }.items()
            if k in stc["variants"]
        }
    for vname, inputs in gb_variants.items():
        for seed in mc["seeds"]:
            t1 = time.time()
            gb = GBDT.fit(data["train"], lgb_params, seed, mc["num_boost_round"], mc["early_stopping_rounds"], inputs)
            fitted_info[f"{vname}_s{seed}"] = {
                "rounds_from_early_stopping": gb.rounds,
                "refit_on_all_training": True,
                "seconds": time.time() - t1,
            }
            for r in preds:
                preds[r][f"{vname}_s{seed}"] = gb.predict(data[r])
            name = f"{vname}_s{seed}"
            A.save_gbdt(gb, models_dir / name)
            reload_check[name] = reload_diff(name, A.load_gbdt(models_dir / name).predict(v2_head()), preds["v2"][name])
    # M3 variants run in a separate PyTorch-only process (experiments/m3_worker.py): this process uses LightGBM,
    # and the two OpenMP runtimes crash or hang when used in one process on this machine.
    m3_coverage = {}
    if use_m3:
        job = run_dir / "m3_job"
        for r in ROLES:
            save_arrays(arrays[r], job / r)
        np.save(job / "y_train.npy", data["train"]["y"].to_numpy().astype(np.float32))
        m3_coverage = {r: arrays[r].coverage for r in ROLES} | {
            "id_vocab_sizes": V.sizes(id_vocab),
            "id_min_count": m3c["id_min_count"],
        }
        arrays.clear()
        (job / "job.json").write_text(
            json.dumps(
                {
                    "variants": m3c["variants"],
                    "seeds": mc["seeds"],
                    "dim": m3c["dim"],
                    "epochs": m3c["epochs"],
                    "batch": m3c["batch"],
                    "lr": m3c["lr"],
                    "device": m3c["device"],
                    "threads": m3c["threads"],
                    "other_sizes": [sizes[c] for c in OTHER_CATS],
                    "id_sizes": [V.sizes(id_vocab)[c] for c in ID_COLS],
                    "models_dir": str(models_dir),
                    "reload_rows": RELOAD_ROWS,
                }
            )
        )
        subprocess.run([sys.executable, "-m", "experiments.m3_worker", str(job)], check=True)
        res = json.loads((job / "result.json").read_text())
        for name, info in res["fitted"].items():
            fitted_info[name] = info
            for r in preds:
                preds[r][name] = np.load(job / "preds" / f"{name}_{r}.npy")
        for name, diff in res["reload_check"].items():  # the worker already checked; re-check here too
            reload_check[name] = require_close(f"reload {name}", diff, RELOAD_TOL)
        for r in ROLES:  # the input arrays are large; predictions and result.json are kept
            shutil.rmtree(job / r)
    timings["models"] = time.time() - t0 - sum(timings.values())

    # 5) calibration on V1, evaluation on V2
    v1, v2 = data["v1"], data["v2"]
    y1 = v1["y"].to_numpy()
    g1_codes, g2_codes = v1[GROUP_KEY].to_numpy(), v2[GROUP_KEY].to_numpy()
    n1, n2 = v1["tr_clicks"].to_numpy(), v2["tr_clicks"].to_numpy()
    lambdas = cfg["calibration"]["lambdas"]
    # model families that can be selected as the base model (seed-matched lists of prediction keys)
    fams = ({"M1": ["M1"]} if "M1" in preds["v2"] else {}) | {"M2": [f"M2_s{s}" for s in mc["seeds"]]}
    if stc.get("enabled"):
        fams |= {v: [f"{v}_s{s}" for s in mc["seeds"]] for v in stc["variants"]}
    if use_m3:
        fams |= {f"M3_{v}": [f"M3_{v}_s{s}" for s in mc["seeds"]] for v in m3c["variants"]}
    base_names = [n for names in fams.values() for n in names]
    g1m = v2["group_class"].to_numpy() == "G1"
    rows: dict[str, dict] = {}
    g1_pred: dict[str, np.ndarray] = {}  # V2 G1-row predictions per "model|calibration" (for paired bootstraps)

    def record(key: str, p: np.ndarray) -> None:
        rows[key] = evaluate(v2, p)
        g1_pred[key] = p[g1m]

    for name in ["M0a", "M0b"] + list(preds["v2"]):
        record(f"{name}|raw", preds["v2"][name])
    cal_info: dict[str, dict] = {}
    cal_objs: dict[str, dict] = {}  # fitted calibrators, kept so the selected ones are saved without refitting
    for name in base_names:
        p1, p2 = preds["v1"][name], preds["v2"][name]
        glob = GlobalPlatt.fit(p1, y1)
        record(f"{name}|ii", glob.predict(p2))
        cal_objs[name] = {"ii": glob, "iii": {}}
        for lam in lambdas:
            gs = GroupShrunkPlatt.fit(p1, y1, g1_codes, glob, lam)
            cal_objs[name]["iii"][lam] = gs
            record(f"{name}|iii_lam{lam:g}", gs.predict(p2, g2_codes))
            cal_info[f"{name}|iii_lam{lam:g}"] = {
                "nonpositive_slopes": gs.n_nonpositive_slopes,
                "converged": gs.converged,
                "groups": int(gs.group_ids.size),
            }
        fa = FreqAware.fit(p1, y1, n1)
        cal_objs[name]["iv"] = fa
        record(f"{name}|iv", fa.predict(p2, n2))
        cal_info[name] = {"ii": {"a": glob.a, "b": glob.b}, "iv": {"w": fa.w.tolist(), "w0": fa.w0}}
    timings["calibration"] = time.time() - t0 - sum(timings.values())

    # 6) selection on V2 only (seed-averaged primary metric), frozen before any test run
    def primary(key: str) -> float:
        return rows[key]["G1"]["macro_logloss"]

    def seed_mean(family: str, suffix: str) -> float:
        return float(np.mean([primary(f"{n}|{suffix}") for n in fams[family]]))

    sc = cfg.get("selection") or {}
    if sc.get("mode") == "model_variant":
        # second study: choose one input variant on V2, all compared after the same (ii) calibration
        base = min(sc["candidates"], key=lambda f: seed_mean(f, "ii"))
    else:
        base = min(fams, key=lambda f: seed_mean(f, "raw"))
    lam = min(lambdas, key=lambda x: seed_mean(base, f"iii_lam{x:g}"))
    method = min([f"iii_lam{lam:g}", "iv"], key=lambda s: seed_mean(base, s))
    if sc.get("mode") == "model_variant":
        ref = sc["reference"]
        primary_cmp = {
            "type": "model",
            "description": f"{base} (ii) vs {ref} (ii)",
            "treatment": [f"{n}|ii" for n in fams[base]],
            "control": [f"{n}|ii" for n in fams[ref]],
        }
    else:
        primary_cmp = {
            "type": "calibration",
            "description": f"{base}: {method} vs (ii)",
            "treatment": [f"{n}|{method}" for n in fams[base]],
            "control": [f"{n}|ii" for n in fams[base]],
        }
    selection = {
        "selected_on": "V2 primary metric (G1 macro LogLoss), averaged over base-model seeds",
        "base_candidates": list(fams),
        "base_model": base,
        "lambda_iii": lam,
        "calibration_method": method,
        "seed_means": {
            f"{f}|{s}": seed_mean(f, s) for f in fams for s in ["raw", "ii", "iv"] + [f"iii_lam{x:g}" for x in lambdas]
        },
        "groups": group_summary["definition"],
        "vocab_fingerprint": V.fingerprint(vocab),
        "id_vocab_fingerprint": V.fingerprint(id_vocab) if use_m3 else None,
        "lgb_params": lgb_params,
        "m3": m3c or None,
        "primary": primary_cmp,
        "selection_mode": sc.get("mode", "calibration"),
    }

    # save the calibrators (with the selected lambda) of every saved base model, then reload-check them
    group_clicks = stats.select(GROUP_KEY, "tr_clicks")
    for name in base_names:
        p2h = preds["v2"][name][:RELOAD_ROWS]
        gl, gsh, fah = cal_objs[name]["ii"], cal_objs[name]["iii"][lam], cal_objs[name]["iv"]
        save_calibrators(gl, gsh, fah, group_clicks, models_dir / f"calib_{name}")
        gl2, gs2, fa2, _ = load_calibrators(models_dir / f"calib_{name}")
        gh, nh = g2_codes[:RELOAD_ROWS], n2[:RELOAD_ROWS]
        # each calibrator is checked on its own (a combined Python max() would hide a NaN)
        for tag, p_re, p_mem in (
            ("ii", gl2.predict(p2h), gl.predict(p2h)),
            ("iii", gs2.predict(p2h, gh), gsh.predict(p2h, gh)),
            ("iv", fa2.predict(p2h, nh), fah.predict(p2h, nh)),
        ):
            require_finite_probs(f"reloaded calib_{name}|{tag}", p_re)
            reload_check[f"calib_{name}|{tag}"] = require_close(
                f"reload calib_{name}|{tag}", float(np.abs(p_re - p_mem).max()), RELOAD_TOL
            )

    # informational paired bootstrap on V2 (selection data, not a confirmatory test). It covers the same
    # seeds as the selection: every seed of the selected family, seed-paired, one user resample per replicate.
    # Models are held fixed, so only the evaluation-user sampling uncertainty is covered.
    bs = cfg["bootstrap"]
    y_g1, g_g1, u_g1 = v2["y"].to_numpy()[g1m], g2_codes[g1m], v2["user_key"].to_numpy()[g1m]
    boot = M.paired_bootstrap_macro_diff(
        y_g1,
        [g1_pred[k] for k in primary_cmp["treatment"]],
        [g1_pred[k] for k in primary_cmp["control"]],
        g_g1,
        u_g1,
        bs["reps"],
        bs["seed"],
    )
    boot["note"] = (
        f"{primary_cmp['description']} ({len(primary_cmp['treatment'])} seed pair(s)) on V2 G1 rows; "
        "selection data, not a test; models fixed (evaluation-user uncertainty only)"
    )

    # purchase-history ablation (raw predictions): same model with vs without history inputs.
    # V2 is not used to choose anything in this comparison; M2 pairs seed k with seed k.
    hist_ablation = {}
    if ablate:
        y_g1, g_g1, u_g1 = v2["y"].to_numpy()[g1m], g2_codes[g1m], v2["user_key"].to_numpy()[g1m]
        hist_ablation["M1"] = M.paired_bootstrap_macro_diff(
            y_g1, preds["v2"]["M1"][g1m], preds["v2"]["M1_nohist"][g1m], g_g1, u_g1, bs["reps"], bs["seed"]
        )
        hist_ablation["M2"] = M.paired_bootstrap_macro_diff(
            y_g1,
            [preds["v2"][f"M2_s{s}"][g1m] for s in mc["seeds"]],
            [preds["v2"][f"M2_nohist_s{s}"][g1m] for s in mc["seeds"]],
            g_g1,
            u_g1,
            bs["reps"],
            bs["seed"],
        )
        for v in hist_ablation.values():
            v["note"] = "with minus without history inputs, raw predictions, V2 G1 rows; negative = history helps"

    # M3 component comparison: attention vs mean pooling with identical inputs, seed-paired
    m3_pooling = {}
    if use_m3 and {"attn", "mean"} <= set(m3c["variants"]):
        m3_pooling = M.paired_bootstrap_macro_diff(
            v2["y"].to_numpy()[g1m],
            [preds["v2"][f"M3_attn_s{s}"][g1m] for s in mc["seeds"]],
            [preds["v2"][f"M3_mean_s{s}"][g1m] for s in mc["seeds"]],
            g2_codes[g1m],
            v2["user_key"].to_numpy()[g1m],
            bs["reps"],
            bs["seed"],
        )
        m3_pooling["note"] = "attention minus mean pooling, raw predictions, V2 G1 rows; negative = attention helps"
    timings["selection_bootstrap"] = time.time() - t0 - sum(timings.values())

    meta = {
        "config": config_path,
        "git": git_at_start,
        "created_utc": run_dir.name,
        "data": {r: [{"file": str(p), "buckets": b, "sha256": sha_of(p)} for p, b in srcs[r]] for r in ROLES},
        "smoke_user_pct": user_pct,
        "rows": {r: data[r].height for r in ROLES},
        "vocab": {"fingerprint": V.fingerprint(vocab), "sizes": sizes, "min_count": cfg["vocab"]["min_count"]},
        "versions": {
            "python": platform.python_version(),
            "polars": pl.__version__,
            "lightgbm": lightgbm.__version__,
            "sklearn": sklearn.__version__,
            "scipy": scipy.__version__,
            "numpy": np.__version__,
        },
        "timings_seconds": {k: round(v, 1) for k, v in timings.items()} | {"total": round(time.time() - t0, 1)},
        "peak_rss_gb": peak_rss_gb(),
        "fitted": fitted_info,
        "calibration": cal_info,
    }
    meta["m3_id_coverage"] = m3_coverage
    meta["stacking"] = stacking_info
    meta["reload_check_max_abs_diff"] = reload_check | {"rows": RELOAD_ROWS}
    out = {
        "meta": meta,
        "groups": group_summary,
        "selection": selection,
        "lgb_search": lgb_search,
        "v2_bootstrap_info": boot,
        "v2_history_ablation": hist_ablation,
        "v2_m3_pooling": m3_pooling,
        "v2_rows": rows,
    }
    (run_dir / "results.json").write_text(json.dumps(out, indent=2, default=str))
    (run_dir / "selection.json").write_text(json.dumps(selection, indent=2, default=str))
    (run_dir / "results.md").write_text(render_markdown(out, mc["seeds"], lambdas))
    print(f"wrote {run_dir}")
    return run_dir


def render_markdown(out: dict, seeds: list[int], lambdas: list[float]) -> str:
    meta, gs, sel, rows = out["meta"], out["groups"], out["selection"], out["v2_rows"]
    d = gs["definition"]
    lines = [
        f"# Results: {Path(meta['config']).stem} (V2 = selection data, not a test)",
        "",
        f"- commit `{meta['git']['commit'][:7]}`" + (" (dirty working tree)" if meta["git"]["dirty_files"] else ""),
        f"- rows: {meta['rows']}, vocab `{meta['vocab']['fingerprint']}`, smoke user % = {meta['smoke_user_pct']}",
        "- data: "
        + "; ".join(
            f"{r}: "
            + ", ".join(
                Path(x["file"]).name + (f" buckets {x['buckets']}" if x["buckets"] else "") for x in meta["data"][r]
            )
            for r in ROLES
        ),
        f"- **G1 groups**: `{d['key']}` with training clicks ≥ {d['n_min']} and training positives < {d['c_min']} "
        f"(selected on training files only): {gs['G1_groups']} groups, {gs['G1_train_click_share']:.2%} of training clicks",
        f"- selection: base **{sel['base_model']}**, λ(iii) = {sel['lambda_iii']:g}, method **{sel['calibration_method']}**; "
        f"primary comparison ({sel.get('selection_mode', 'calibration')}): **{sel['primary']['description']}**",
        f"- time {meta['timings_seconds']['total']} s, peak memory {meta['peak_rss_gb']} GB",
        "",
        "| model | calib | G1 macro LogLoss | G1 pooled O/E | G1 median abs(O/E−1) | G1 within-group AUC | LogLoss (non-missing) | AUC (non-missing) | GAUC (non-missing) | O/E (non-missing) | ECE10 | AUC (all rows) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    families = {
        "M0a": ["M0a"],
        "M0b": ["M0b"],
        "M1": ["M1"],
        "M1_nohist": ["M1_nohist"],
        "M2": [f"M2_s{s}" for s in seeds],
        "M2_nohist": [f"M2_nohist_s{s}" for s in seeds],
        "M2aux": [f"M2aux_s{s}" for s in seeds],
        "M2te": [f"M2te_s{s}" for s in seeds],
        "M2both": [f"M2both_s{s}" for s in seeds],
        "M3_attn": [f"M3_attn_s{s}" for s in seeds],
        "M3_mean": [f"M3_mean_s{s}" for s in seeds],
    }
    suffixes = ["raw", "ii"] + [f"iii_lam{x:g}" for x in lambdas] + ["iv"]
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

            lines.append(
                f"| {fam} | {suf} | {cell(('G1', 'macro_logloss'))} | {cell(('G1', 'pooled_O_over_E'), '.3f')} | "
                f"{cell(('G1', 'median_abs_ratio_dev'), '.3f')} | {cell(('G1', 'within_group_auc_macro'), '.4f')} | "
                f"{cell(('overall_nonmissing', 'logloss'))} | {cell(('overall_nonmissing', 'auc'), '.4f')} | "
                f"{cell(('overall_nonmissing', 'gauc'), '.4f')} | "
                f"{cell(('overall_nonmissing', 'O_over_E'), '.3f')} | {cell(('overall_nonmissing', 'ece10'), '.5f')} | "
                f"{cell(('overall_all', 'auc'), '.4f')} |"
            )
    b = out["v2_bootstrap_info"]
    lines += [
        "",
        f"Paired user bootstrap on V2 G1 rows ({b['note']}): diff = {b['diff']:+.6f}, "
        f"95% CI [{b['ci95'][0]:+.6f}, {b['ci95'][1]:+.6f}], {b['reps']} reps, {b['users']} users, {b['groups']} groups.",
    ]
    for fam, h in out.get("v2_history_ablation", {}).items():
        lines.append(
            f"History ablation {fam} ({h['note']}): diff = {h['diff']:+.6f}, 95% CI "
            f"[{h['ci95'][0]:+.6f}, {h['ci95'][1]:+.6f}], seed pairs {h['seed_pairs']}, "
            f"per pair {[round(x, 6) for x in h['per_pair_diff']]}."
        )
    mp = out.get("v2_m3_pooling") or {}
    if mp:
        lines.append(
            f"M3 pooling comparison ({mp['note']}): diff = {mp['diff']:+.6f}, 95% CI "
            f"[{mp['ci95'][0]:+.6f}, {mp['ci95'][1]:+.6f}], seed pairs {mp['seed_pairs']}, "
            f"per pair {[round(x, 6) for x in mp['per_pair_diff']]}."
        )
    cov = meta.get("m3_id_coverage") or {}
    if cov:
        v2c = cov.get("v2", {})
        v2c = v2c.get("per_file", [v2c])[0] if isinstance(v2c, dict) else {}
        lines.append(
            f"M3 id vocabulary (min_count {cov['id_min_count']}): sizes "
            f"{ {k: cov['id_vocab_sizes'][k] for k in ('ad_feat0', 'ad_feat1', 'ad_feat2')} }; V2 unknown shares "
            f"{ {k: round(v['target_unknown_share'], 4) for k, v in v2c.items()} } (targets)."
        )
    ls = out.get("lgb_search") or {}
    if ls:
        lines += ["", f"LightGBM candidates ({ls['rule']}; chosen **{ls['chosen']}**):"]
        lines += [
            f"- {k}: params {v['params']}, rounds {v['rounds']}, V2 G1 macro LogLoss {v['v2_G1_macro_logloss']:.5f}, "
            f"AUC {v['v2_auc_nonmissing']:.4f}"
            for k, v in ls["candidates"].items()
        ]
    se = gs.get("sensitivity_expected_positives") or {}
    if se:
        lo, hi = se["G1exp_click_range"]
        lines += [
            "",
            f"Sensitivity G1exp (expected positives = training clicks × p0 = {se['p0']:.4%} < {d['c_min']}, "
            f"i.e. {lo} ≤ training clicks < {hi:.0f}): {se['G1exp_groups']} groups, "
            f"{se['G1exp_train_click_share']:.2%} of training clicks, overlap with G1 {se['G1_and_G1exp_overlap']} groups.",
            "",
            "| model | calib | G1exp macro LogLoss | G1exp pooled O/E |",
            "|---|---|---|---|",
        ]
        for fam, names in families.items():
            for suf in suffixes:
                keys = [f"{n}|{suf}" for n in names if f"{n}|{suf}" in rows and "G1exp" in rows[f"{n}|{suf}"]]
                if keys:
                    ll = mean_sd([rows[k]["G1exp"]["macro_logloss"] for k in keys])
                    oe = mean_sd([rows[k]["G1exp"]["pooled_O_over_E"] for k in keys])
                    lines.append(
                        f"| {fam} | {suf} | {ll['mean']:.5f}"
                        + (f" ± {ll['sd']:.5f}" if ll["n"] > 1 else "")
                        + f" | {oe['mean']:.3f} |"
                    )
    gx = next(iter(rows.values()))["overall_nonmissing"]
    lines += [
        "",
        f"GAUC: per-user AUC weighted by clicks over users with both classes "
        f"({gx['users_used']} users used, {gx['users_excluded']} excluded, {gx['click_share_used']:.1%} of non-missing clicks).",
        "Values with ± are mean ± sd over base-model seeds.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main(sys.argv[1])
