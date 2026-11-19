"""Mechanism study B (docs/DESIGN.md §15.2): does the second study's auxiliary input help because of the other
attribution label (mta), or because one more out-of-fold prediction is added as an input?

Control: the same out-of-fold procedure with the main label (last > 0) -> input "aux_last".
Development data only (train-00..02, V1 train-03, V2 train-04 of the second study's development run).

    PYTHONPATH=. uv run python scripts/aux_label_control.py runs/p2_train00_02/20261117T131443Z runs/aux_control \
        --md docs/results/aux_label_control.md
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from cvr import groups as G
from cvr import metrics as M
from cvr.artifacts import load_gbdt, load_stacking, save_gbdt, save_stacking
from cvr.calibration import GlobalPlatt
from cvr.features import featurize, read_raw
from cvr.final_eval import verify_freeze, vocab_frames
from cvr.models import GBDT
from cvr.schema import CAT_INPUTS, GROUP_KEY, NUM_INPUTS
from cvr.stacking import AUX_LABEL, TE_INPUTS, fit_stacking

BASE = CAT_INPUTS + NUM_INPUTS
NEW = {"M2auxlast": BASE + ["aux_last"], "M2bothlast": BASE + ["aux_last"] + TE_INPUTS}
STORED = ["M2", "M2aux", "M2te", "M2both"]
COMPARISONS = [("M2aux", "M2auxlast", "core: same procedure, label mta vs last"),
               ("M2both", "M2bothlast", "core: same procedure, label mta vs last (with group rates)"),
               ("M2auxlast", "M2", "reference: adding a same-label out-of-fold prediction"),
               ("M2aux", "M2", "reference: the second study's mta input")]
REPRO_TOL, IDENTITY_TOL, TE_TOL = 1e-9, 1e-12, 1e-9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--md", type=Path, default=None)
    a = ap.parse_args()
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = verify_freeze(a.run)
    cfg, seeds = manifest["config"], manifest["seeds"]
    mc, stc, bs = cfg["models"], cfg["models"]["stacking"], cfg["bootstrap"]
    lgb_params = mc["lgb"]  # fixed from the first study; the second study ran no candidate search
    vocab = vocab_frames(a.run / "vocab")
    stats = pl.read_parquet(a.run / "frozen_group_stats.parquet")
    gc, p0 = manifest["groups"]["definition"], manifest["groups"]["p0_train"]
    root = Path(cfg["data"]["root"])
    data = {}
    for role in ("train", "v1", "v2"):
        df = pl.concat([featurize(read_raw(root / f), vocab) for f in cfg["data"][role]])
        data[role] = pl.concat([df, G.classify(df, stats, gc["n_min"], gc["c_min"], gc["c_dense"], p0)], how="horizontal")
    log = {"rows": {r: data[r].height for r in data}, "seconds": {"features": round(time.time() - t0)}}
    print(f"features {log['rows']} in {time.time() - t0:.0f}s", flush=True)

    # check 1: the rebuilt training rows give the saved full auxiliary model exactly (same data, same procedure)
    saved = load_stacking(a.run / "models" / "stacking")
    refit = GBDT.fit(data["train"], lgb_params, stc["aux_seed"], mc["num_boost_round"], mc["early_stopping_rounds"],
                     BASE, label=AUX_LABEL)
    head = data["v2"].head(100_000)
    log["identity_check"] = {"refit_rounds": refit.rounds, "saved_rounds": saved.aux.rounds,
                             "max_abs_pred_diff": float(np.abs(refit.predict(head) - saved.aux.predict(head)).max())}
    if refit.rounds != saved.aux.rounds or log["identity_check"]["max_abs_pred_diff"] > IDENTITY_TOL:
        raise SystemExit(f"training data or procedure differs from the saved run: {log['identity_check']}")
    print(f"identity ok {log['identity_check']}", flush=True)

    # the control input: same out-of-fold procedure, main label
    t1 = time.time()
    train_last, st_last, info = fit_stacking(data["train"], lgb_params, stc["aux_seed"], stc["folds"],
                                             mc["num_boost_round"], mc["early_stopping_rounds"], True, True,
                                             aux_label="y", aux_col="aux_last")
    save_stacking(st_last, a.out / "stacking_last")
    log["stacking_last"] = info | {"seconds": round(time.time() - t1)}
    frames = {}
    for r in ("v1", "v2"):
        f = saved.transform(data[r])
        last = st_last.transform(data[r])
        for c in TE_INPUTS:  # group-rate inputs come from the same training rows: equal up to the saved run's
            # floating-point drift (its prior strength was fitted with an unordered group_by; docs/DESIGN.md §15.2)
            d = float(np.abs(f[c].to_numpy() - last[c].to_numpy()).max())
            log.setdefault("group_rate_max_abs_diff", {})[f"{r}:{c}"] = d
            if d > TE_TOL:
                raise SystemExit(f"group-rate input {c} differs between the saved run and the control: {d}")
        frames[r] = f.with_columns(last["aux_last"])
    preds = {r: {} for r in frames}
    for v in STORED:
        for s in seeds:
            m = load_gbdt(a.run / "models" / f"{v}_s{s}")
            for r in frames:
                preds[r][f"{v}_s{s}"] = m.predict(frames[r])
    for v, inputs in NEW.items():
        for s in seeds:
            t1 = time.time()
            m = GBDT.fit(train_last, lgb_params, s, mc["num_boost_round"], mc["early_stopping_rounds"], inputs)
            save_gbdt(m, a.out / f"{v}_s{s}")
            log.setdefault("fitted", {})[f"{v}_s{s}"] = {"rounds": m.rounds, "seconds": round(time.time() - t1)}
            for r in frames:
                preds[r][f"{v}_s{s}"] = m.predict(frames[r])
            print(f"{v}_s{s}: {m.rounds} rounds", flush=True)

    # (ii) global calibration on V1, evaluation on V2 G1 (as in §13)
    v1, v2 = frames["v1"], frames["v2"]
    y1, y2 = v1["y"].to_numpy(), v2["y"].to_numpy()
    g1m = v2["group_class"].to_numpy() == "G1"
    g2, users = v2[GROUP_KEY].to_numpy()[g1m], v2["user_key"].to_numpy()[g1m]
    cal = {k: GlobalPlatt.fit(preds["v1"][k], y1).predict(preds["v2"][k])[g1m] for k in preds["v2"]}
    macro = {k: M.group_report(y2[g1m], p, g2)["macro_logloss"] for k, p in cal.items()}
    stored = json.loads((a.run / "results.json").read_text())["v2_rows"]
    repro = {}
    for v in STORED:
        for s in seeds:
            k = f"{v}_s{s}"
            ref = stored[f"{k}|ii"]["G1"]["macro_logloss"]
            if abs(macro[k] - ref) > REPRO_TOL:
                raise SystemExit(f"reproduction failed for {k}: {macro[k]} vs stored {ref}")
            repro[k] = {"recomputed": macro[k], "stored": ref}
    variants = STORED + list(NEW)
    seed_mean = {v: float(np.mean([macro[f"{v}_s{s}"] for s in seeds])) for v in variants}
    comps = []
    for va, vb, role in COMPARISONS:
        b = M.paired_bootstrap_macro_diff(y2[g1m], [cal[f"{va}_s{s}"] for s in seeds],
                                          [cal[f"{vb}_s{s}"] for s in seeds], g2, users, bs["reps"], bs["seed"])
        comps.append({"a": va, "b": vb, "role": role} | b)
    result = {"run": str(a.run), "seeds": seeds, "design": "docs/DESIGN.md §15.2", "log": log, "reproduction": repro,
              "v2_G1_macro_logloss_ii_seed_mean": seed_mean, "comparisons": comps,
              "seconds_total": round(time.time() - t0)}
    (a.out / "results.json").write_text(json.dumps(result, indent=1))
    if a.md:
        write_md(a.md, result)
        a.md.with_suffix(".json").write_text(json.dumps(result, indent=1))
    print(f"done in {time.time() - t0:.0f}s")


def write_md(path: Path, r: dict) -> None:
    lines = ["# Study B: auxiliary attribution label vs a same-label control (development data)", "",
             f"Run `{r['run']}` (train-00..02, V1 train-03, V2 train-04); seeds {r['seeds']}. Design: {r['design']}. "
             "All variants after (ii) global calibration fitted on V1; metric V2 G1 macro LogLoss. "
             "M2both was chosen on this V2, so comparisons with M2both are optimistic.", "",
             f"Checks: refit of the saved mta auxiliary model on the rebuilt training rows: {r['log']['identity_check']}; "
             f"{len(r['reproduction'])} stored V2 values recomputed within {REPRO_TOL:g}; group-rate inputs equal up to "
             f"{max(r['log']['group_rate_max_abs_diff'].values()):.1e} (saved run's floating-point drift).", "",
             "| variant | inputs added to M2 | V2 G1 macro LogLoss (ii), seed mean |", "|---|---|---:|"]
    added = {"M2": "-", "M2aux": "out-of-fold prediction of mta > 0", "M2te": "group rates",
             "M2both": "mta prediction + group rates", "M2auxlast": "out-of-fold prediction of last > 0 (control)",
             "M2bothlast": "last prediction + group rates (control)"}
    for v, x in r["v2_G1_macro_logloss_ii_seed_mean"].items():
        lines.append(f"| {v} | {added[v]} | {x:.6f} |")
    lines += ["", "| comparison | role | difference A - B [95% CI] | per seed pair |", "|---|---|---|---|"]
    for c in r["comparisons"]:
        pairs = ", ".join(f"{d:+.6f}" for d in c["per_pair_diff"])
        lines.append(f"| {c['a']} - {c['b']} | {c['role']} | {c['diff']:+.6f} [{c['ci95'][0]:+.6f}, "
                     f"{c['ci95'][1]:+.6f}] | {pairs} |")
    lines += ["", "Intervals resample V2 users with the fitted models held fixed (seed-paired); they do not include "
              "training variation."]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
