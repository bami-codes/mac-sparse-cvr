"""Study D (docs/DESIGN.md §17): intercept-only group shrinkage (iii-0) vs intercept+slope (iii-d) vs (ii),
a backtest on files that were all opened before (pre-registered in §17 before any computation).

    PYTHONPATH=. uv run python scripts/study_d.py runs/c_dev_train08_10/<stamp> runs/dev_train15_17_m3/<stamp> runs/study_d

Setting C (primary): study C's small context groups, G-int (and G-shared) seeds; V1 train-11, V2 train-12, evaluation
train-13 + train-14. Setting 1 (secondary): the first study's G1 groups, M2 seeds; V1 train-18, V2 train-19,
evaluation the official test file. Base models are NOT retrained; their predictions on the evaluation files are
recomputed from the saved models and must reproduce the recorded (ii) / (iii-d) metrics (hard check) before
anything new is computed.
"""

import json
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from cvr import domain_eval as D
from cvr import groups as G
from cvr import metrics as M
from cvr.artifacts import load_gbdt
from cvr.calibration import GlobalPlatt, fit_group_intercept_exact, fit_group_shrunk_exact
from cvr.checks import require_close
from cvr.domain_models import load_gbdt_offset
from cvr.features import featurize, read_raw
from cvr.schema import DOMAIN_KEY, GROUP_KEY, TARGET

LAMBDAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
SEEDS = [2026, 2027, 2028]
REPS, BOOT_SEED = 1000, 20261118
RETAIN_MARGIN = 5e-5  # §17.4
TOL = 1e-9


def vocab(run: Path) -> dict[str, pl.DataFrame]:
    return {p.stem: pl.read_parquet(p) for p in (run / "vocab").glob("*.parquet")}


def labels(path: Path) -> np.ndarray:
    return (pl.read_parquet(path, columns=[TARGET])[TARGET].to_numpy() > 0).astype(np.int8)


def group_metrics(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> dict:
    pg = M.per_group(y, p, g)
    return {"macro": float(pg["logloss"].mean()), "click": float(M.bce(y, p).mean()),
            "poisson": float(D.poisson_deviance(pg["O"], pg["E"]).mean()), "pooled_O_over_E": float(y.sum() / p.sum()),
            "groups": pg.height, "rows": int(y.size)}


def fit_all(p1: np.ndarray, y1: np.ndarray, g1: np.ndarray) -> dict:
    glob = GlobalPlatt.fit(p1, y1)
    out = {"ii": glob, "iiid": {}, "iii0": {}}
    for lam in LAMBDAS:
        gs, i1 = fit_group_shrunk_exact(p1, y1, g1, glob, lam)
        g0, i0 = fit_group_intercept_exact(p1, y1, g1, glob, lam)
        if not (gs.converged and g0.converged):
            raise RuntimeError(f"calibration did not converge at lambda {lam}: {i1} {i0}")
        out["iiid"][lam], out["iii0"][lam] = gs, g0
    return out


def apply(cal: dict, method: str, lam: float | None, p: np.ndarray, g: np.ndarray) -> np.ndarray:
    return cal["ii"].predict(p) if method == "ii" else cal[method][lam].predict(p, g)


def run_setting(name: str, fams: dict, data: dict, eval_mask: dict, groups: dict, users: np.ndarray,
                file_of: np.ndarray | None) -> dict:
    """fams: family -> {seed: {"v1": p, "v2": p, "ev": p}}; data: role -> y; groups: role -> codes (all rows);
    eval_mask: role -> rows of the evaluated groups (small / G1)."""
    out = {}
    for fam, by_seed in fams.items():
        cals = {s: fit_all(by_seed[s]["v1"], data["v1"], groups["v1"]) for s in SEEDS}
        m2, me = eval_mask["v2"], eval_mask["ev"]

        def seed_metric(role: str, method: str, lam: float | None, key: str) -> float:
            mask = m2 if role == "v2" else me
            return float(np.mean([group_metrics(data[role][mask], apply(cals[s], method, lam, by_seed[s][role], groups[role])[mask],
                                                groups[role][mask])[key] for s in SEEDS]))

        sel = {"main": {}, "aggregate": {}}
        for method in ("iiid", "iii0"):
            sel["main"][method] = min(LAMBDAS, key=lambda l: seed_metric("v2", method, l, "macro"))
            sel["aggregate"][method] = min(LAMBDAS, key=lambda l: seed_metric("v2", method, l, "poisson"))
        v2_table = {f"{m}|lam{l:g}": {k: seed_metric("v2", m, l, k) for k in ("macro", "click", "poisson", "pooled_O_over_E")}
                    for m in ("iiid", "iii0") for l in LAMBDAS} | {
            "ii": {k: seed_metric("v2", "ii", None, k) for k in ("macro", "click", "poisson", "pooled_O_over_E")}}

        def preds(method: str, rule: str = "main") -> list[np.ndarray]:
            lam = None if method == "ii" else sel[rule][method]
            return [apply(cals[s], method, lam, by_seed[s]["ev"], groups["ev"])[me] for s in SEEDS]

        ye, ge, ue = data["ev"][me], groups["ev"][me], users[me]
        ev_table = {}
        for rule in ("main", "aggregate"):
            for method in ("ii", "iiid", "iii0"):
                ps = preds(method, rule)
                ev_table[f"{rule}|{method}"] = {k: float(np.mean([group_metrics(ye, q, ge)[k] for q in ps]))
                                                for k in ("macro", "click", "poisson", "pooled_O_over_E")}

        def boot(a: str, b: str, mask: np.ndarray | None = None) -> dict:
            pa, pb = preds(a), preds(b)
            mm = np.ones(ye.size, bool) if mask is None else mask
            return D.paired_bootstrap_groups(ye[mm], [x[mm] for x in pa], [x[mm] for x in pb], ge[mm], ue[mm],
                                             REPS, BOOT_SEED, pair_seeds=SEEDS)

        comps = {"iii0 - iiid": boot("iii0", "iiid"), "iii0 - ii": boot("iii0", "ii"), "iiid - ii": boot("iiid", "ii")}
        per_file = {}
        if file_of is not None:
            fo = file_of[me]
            per_file = {str(k): boot("iii0", "iiid", fo == k) for k in np.unique(fo)}
        out[fam] = {"lambda": sel, "v2": v2_table, "eval": ev_table, "comparisons": comps, "per_file_iii0_minus_iiid": per_file}
        print(name, fam, "lambda", sel, "iii0-iiid", {k: round(v, 7) for k, v in comps["iii0 - iiid"]["point"].items()}, flush=True)
    return out


def verdict(c: dict) -> dict:
    b = c["comparisons"]["iii0 - iiid"]
    retain = bool(b["ci95"]["macro"][1] < RETAIN_MARGIN)
    reduce = bool(b["point"]["poisson"] < 0 and b["ci95"]["poisson"][1] < 0 and all(x < 0 for x in b["per_pair"]["poisson"]))
    return {"logloss_gain_retained": retain, "group_count_error_reduced": reduce,
            "rule": f"retain: dLL 95% CI upper < {RETAIN_MARGIN}; reduce: dPD < 0, CI upper < 0, all 3 seed pairs < 0"}


def setting_c(run: Path) -> tuple[dict, dict]:
    res = json.loads((run / "results.json").read_text())
    test = json.loads((run / "final" / "test" / "results.json").read_text())
    sel = json.loads((run / "selection.json").read_text())
    small = set(sel["domains"]["small"])
    meta = res["meta"]["data"]
    v1f, v2f = Path(meta["v1"][0]["file"]), Path(meta["v2"][0]["file"])
    from cvr.mddata import DOM_COL, load_md
    dom = {r: np.asarray(load_md(run / "nn_data" / r).cat[:, DOM_COL]) for r in ("v1", "v2")}
    y = {"v1": labels(v1f), "v2": labels(v2f)}
    voc = vocab(run)
    parts = [featurize(read_raw(Path(f)), voc) for f in test["files"]]
    ev = pl.concat(parts)
    file_of = np.concatenate([np.full(p.height, k) for k, p in enumerate(parts)])
    del parts
    y["ev"], dom["ev"] = ev["y"].to_numpy(), ev[DOMAIN_KEY].to_numpy()
    users = ev["user_key"].to_numpy()
    fams = {}
    for fam, loader in (("G-int", load_gbdt_offset), ("G-shared", load_gbdt)):
        fams[fam] = {s: {"v1": np.load(run / "preds" / f"{fam}_s{s}_v1.npy"), "v2": np.load(run / "preds" / f"{fam}_s{s}_v2.npy"),
                         "ev": loader(run / "models" / f"{fam}_s{s}").predict(ev)} for s in SEEDS}
    mask = {r: np.isin(dom[r], list(small)) for r in ("v1", "v2", "ev")}
    # reproduction checks: (ii) and (iii-d, lambda 1) on V2 and on the evaluation files equal study C's records
    checked = 0
    for fam in fams:
        for s in SEEDS:
            cal = fit_all(fams[fam][s]["v1"], y["v1"], dom["v1"])
            for role, rec in (("v2", res["v2_rows"]), ("ev", test["metrics"])):
                for method, key in (("ii", "ii"), ("iiid", "iiid_lam1")):
                    q = apply(cal, method, 1.0, fams[fam][s][role], dom[role])
                    got = group_metrics(y[role][mask[role]], q[mask[role]], dom[role][mask[role]])["macro"]
                    require_close(f"setting C {fam} s{s} {role} {key}", abs(got - rec[f"{fam}_s{s}|{key}"]["small"]["macro_logloss"]), TOL)
                    checked += 1
    out = run_setting("C", fams, y, mask, dom, users, file_of)
    return out, {"reproduced_values": checked, "eval_rows": int(ev.height), "files": test["files"]}


def setting_1(run: Path) -> tuple[dict, dict]:
    cfg = tomllib.loads((run / "config.toml").read_text())
    res = json.loads((run / "results.json").read_text())
    test = json.loads((run / "final" / "test" / "results.json").read_text())
    voc = vocab(run)
    root = Path(cfg["data"]["root"])
    files = {"v1": root / cfg["data"]["v1"][0], "v2": root / cfg["data"]["v2"][0], "ev": Path(test["eval_file"])}
    stats = pl.read_parquet(run / "frozen_group_stats.parquet")
    gc = cfg["groups"]
    y, grp, g1, users, preds = {}, {}, {}, None, {s: {} for s in SEEDS}
    models = {s: load_gbdt(run / "models" / f"M2_s{s}") for s in SEEDS}
    for role, f in files.items():
        df = featurize(read_raw(f), voc)
        y[role], grp[role] = df["y"].to_numpy(), df[GROUP_KEY].to_numpy()
        g1[role] = G.classify(df, stats, gc["n_min"], gc["c_min"], gc["c_dense"])["group_class"].to_numpy() == "G1"
        if role == "ev":
            users = df["user_key"].to_numpy()
        for s in SEEDS:
            preds[s][role] = models[s].predict(df)
        del df
    checked = 0
    for s in SEEDS:
        cal = fit_all(preds[s]["v1"], y["v1"], grp["v1"])
        for role, rec in (("v2", res["v2_rows"]), ("ev", test["metrics"])):
            q = apply(cal, "ii", None, preds[s][role], grp[role])
            got = group_metrics(y[role][g1[role]], q[g1[role]], grp[role][g1[role]])["macro"]
            require_close(f"setting 1 M2 s{s} {role} ii", abs(got - rec[f"M2_s{s}|ii"]["G1"]["macro_logloss"]), TOL)
            checked += 1
    out = run_setting("1", {"M2": preds}, y, g1, grp, users, None)
    return out, {"reproduced_values": checked, "eval_rows": int(y["ev"].size)}


def render(r: dict) -> str:
    lines = ["# Study D: intercept-only group shrinkage (iii-0) vs intercept+slope (iii-d) vs (ii)", "",
             "Backtest on files opened before (docs/DESIGN.md §17); not a confirmation. Differences are A - B; "
             "seed-paired user bootstrap, models and calibrators fixed.", "",
             f"- verdict (setting C, G-int, train-13 + 14): {r['verdict']}", ""]
    for sname, s in r["settings"].items():
        for fam, f in s.items():
            lines += [f"## Setting {sname}, {fam}", "", f"- lambda (main rule = V2 macro LogLoss): {f['lambda']['main']}; "
                      f"aggregate rule (V2 Poisson deviance): {f['lambda']['aggregate']}", "",
                      "| rule | method | macro LogLoss | click LogLoss | per-group Poisson deviance | pooled O/E |",
                      "|---|---|---:|---:|---:|---:|"]
            for k, v in f["eval"].items():
                rule, method = k.split("|")
                lines.append(f"| {rule} | {method} | {v['macro']:.6f} | {v['click']:.6f} | {v['poisson']:.4f} | {v['pooled_O_over_E']:.3f} |")
            lines += ["", "| comparison (main lambda) | macro [95% CI] | click [95% CI] | Poisson deviance [95% CI] | per seed pair (Poisson) |",
                      "|---|---|---|---|---|"]
            for k, b in f["comparisons"].items():
                p, c = b["point"], b["ci95"]
                lines.append(f"| {k} | {p['macro']:+.6f} [{c['macro'][0]:+.6f}, {c['macro'][1]:+.6f}] | "
                             f"{p['click']:+.6f} [{c['click'][0]:+.6f}, {c['click'][1]:+.6f}] | "
                             f"{p['poisson']:+.4f} [{c['poisson'][0]:+.4f}, {c['poisson'][1]:+.4f}] | "
                             f"{[round(x, 4) for x in b['per_pair']['poisson']]} |")
            for fk, b in f["per_file_iii0_minus_iiid"].items():
                p, c = b["point"], b["ci95"]
                lines.append(f"| iii0 - iiid, file {fk} | {p['macro']:+.6f} [{c['macro'][0]:+.6f}, {c['macro'][1]:+.6f}] | "
                             f"{p['click']:+.6f} | {p['poisson']:+.4f} [{c['poisson'][0]:+.4f}, {c['poisson'][1]:+.4f}] | |")
            lines.append("")
    return "\n".join(lines) + "\n"


def main(run_c: str, run_1: str, out_root: str) -> None:
    t0 = time.time()
    import subprocess
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.splitlines()
    sc, mc = setting_c(Path(run_c))
    s1, m1 = setting_1(Path(run_1))
    r = {"git": {"commit": git, "dirty_files": dirty}, "runs": {"C": run_c, "1": run_1},
         "checks": {"C": mc, "1": m1}, "settings": {"C": sc, "1": s1}, "verdict": verdict(sc["G-int"]),
         "seconds": round(time.time() - t0, 1)}
    out = Path(out_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    (out / "results.json").write_text(json.dumps(r, indent=2, default=str))
    (out / "results.md").write_text(render(r))
    print(f"wrote {out}; verdict {r['verdict']}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
