"""Two real-data diagnostics for study C (docs/DESIGN.md §16 revision 4), from a finished development run.

1. Calibration solver: refit (iii-d) on V1 with the OLD joint L-BFGS-B (GroupShrunkPlatt.fit) and with the exact
   per-domain Newton solve, for every base model and lambda; report how much the V2 small-domain macro LogLoss
   moves, and choose every lambda and the candidate AGAIN from the exact scores. The diagnostic FAILS if that
   independent choice, or the recomputed scores, differ from the run's selection.json (review: it used to print
   the stored choice back). It also reports the global Platt slopes: the revision-4 solver started from an
   infeasible point when a slope was below MIN_SLOPE.
2. STAR domain-ID dimensions: for every saved N-star model, count the (domain, ID-embedding dimension) pairs whose
   running variance passes the 1% rule, i.e. that would have used the domain's own statistics at evaluation under
   the rule before revision 4 (the run itself always uses the global statistics there).

Only development files are read (V1 / V2 labels from the run's own files). The two parts run as separate commands so
PyTorch never shares a process with scikit-learn (OpenMP runtimes); LightGBM is never imported.

    PYTHONPATH=. uv run python scripts/study_c_diagnostics.py calib runs/c_dev_train08_10/<run>
    PYTHONPATH=. uv run python scripts/study_c_diagnostics.py star runs/c_dev_train08_10/<run>
Outputs: <run>/diagnostics_calib.json, <run>/diagnostics_star.json
"""

import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import polars as pl

from cvr import domain_eval as D
from cvr.mddata import DOM_COL, load_md
from cvr.schema import DOMAIN_KEY, TARGET


def labels(run: Path, role: str) -> np.ndarray:
    """y of the run's rows, read with the same file / bucket / user filters (cvr.features.read_raw)."""
    from cvr.features import read_raw

    meta = json.loads((run / "results.json").read_text())["meta"]
    parts = [read_raw(Path(x["file"]), tuple(x["buckets"]) if x["buckets"] else None, meta["smoke_user_pct"],
                      columns=[TARGET])[TARGET].to_numpy() > 0 for x in meta["data"][role]]
    y = np.concatenate(parts).astype(np.int8)
    if y.size != meta["rows"][role]:
        raise ValueError(f"{role}: {y.size} labels for {meta['rows'][role]} rows")
    return y


def base_pred(run: Path, name: str, role: str) -> np.ndarray:
    p = run / "preds" / f"{name}_{role}.npy"
    if p.exists():
        return np.load(p)
    saved = next((run / "nn_final" / "preds").glob(f"{name}_e*_{role}.npy"))
    return np.load(saved)


def calib(run: Path) -> dict:
    from cvr import metrics as M
    from cvr.calibration import MIN_SLOPE, GlobalPlatt, GroupShrunkPlatt, fit_group_shrunk_exact

    res = json.loads((run / "results.json").read_text())
    sel, seeds = res["selection"], res["selection"]["seeds"]
    lambdas = tomllib.loads((run / "config.toml").read_text())["calibration"]["lambdas"]
    cfg_dom = res["domains"]["definition"]
    tr = load_md(run / "nn_data" / "train")
    train = pl.DataFrame({DOMAIN_KEY: np.asarray(tr.cat[:, DOM_COL]), "y": np.load(run / "nn_data" / "y_train.npy")})
    stats = D.domain_stats(train)
    dom = {r: np.asarray(load_md(run / "nn_data" / r).cat[:, DOM_COL]) for r in ("v1", "v2")}
    y = {r: labels(run, r) for r in ("v1", "v2")}
    cls2 = D.domain_class(pl.DataFrame({DOMAIN_KEY: dom["v2"]}), stats, cfg_dom["small_share"], cfg_dom["small_min_clicks"])
    small = cls2 == "small"

    def small_macro(p2: np.ndarray) -> float:
        return float(M.per_group(y["v2"][small], p2[small], dom["v2"][small])["logloss"].mean())

    families = ["G-int", "G-shared", "N-int", "N-shared", "N-head", "N-star", "N-star-noPN"]
    per, old_lam_scores, new_lam_scores, slopes = {}, {}, {}, {}
    for fam in families:
        old_lam_scores[fam], new_lam_scores[fam] = {}, {}
        for lam in lambdas:
            old_s, new_s = [], []
            for s in seeds:
                name = f"{fam}_s{s}"
                p1, p2 = base_pred(run, name, "v1"), base_pred(run, name, "v2")
                glob = GlobalPlatt.fit(p1, y["v1"])
                slopes[name] = glob.a
                old = GroupShrunkPlatt.fit(p1, y["v1"], dom["v1"], glob, lam)
                new, info = fit_group_shrunk_exact(p1, y["v1"], dom["v1"], glob, lam)
                if not new.converged:
                    raise RuntimeError(f"exact (iii-d) did not converge for {name}, lambda {lam}: {info}")
                a, b = small_macro(old.predict(p2, dom["v2"])), small_macro(new.predict(p2, dom["v2"]))
                old_s.append(a)
                new_s.append(b)
                per[f"{name}|lam{lam:g}"] = {"old": a, "exact": b, "exact_minus_old": b - a}
            old_lam_scores[fam][lam], new_lam_scores[fam][lam] = float(np.mean(old_s)), float(np.mean(new_s))
    run_scores = sel["candidate_scores"]
    old_lam = {f: min(lambdas, key=lambda l: old_lam_scores[f][l]) for f in families}
    old_scores = {f: old_lam_scores[f][old_lam[f]] for f in D.PREFERENCE}
    # independent re-selection from the exact scores (same rules as cvr/study_c.py), compared with the run
    new_lam = {f: min(lambdas, key=lambda l: new_lam_scores[f][l]) for f in families}
    new_scores = {f: new_lam_scores[f][new_lam[f]] for f in D.PREFERENCE}
    new_cand = D.choose_candidate(new_scores)
    stored_lam = {f: float(v) for f, v in sel["lambda_iiid"].items()}
    score_gap = max(abs(new_scores[f] - run_scores[f]) for f in D.PREFERENCE)
    problems = []
    if new_lam != stored_lam:
        problems.append(f"lambda: recomputed {new_lam} vs stored {stored_lam}")
    if new_cand != sel["candidate"]:
        problems.append(f"candidate: recomputed {new_cand} vs stored {sel['candidate']}")
    if score_gap > 1e-12:
        problems.append(f"candidate scores differ by up to {score_gap:.3e}")
    if problems:
        raise RuntimeError("the run's selection is not reproduced: " + "; ".join(problems))
    diffs = [v["exact_minus_old"] for v in per.values()]
    part1 = {"lambdas": lambdas,
             "exact_reproduces_run_scores_max_abs": score_gap,
             "global_platt_slope": {"min": min(slopes.values()), "max": max(slopes.values()),
                                    "below_MIN_SLOPE": sorted(n for n, a in slopes.items() if a < MIN_SLOPE)},
             "exact_minus_old_small_macro": {"max_abs_single_seed": float(np.max(np.abs(diffs))),
                                             "max_abs_seed_mean": max(abs(new_lam_scores[f][l] - old_lam_scores[f][l])
                                                                      for f in families for l in lambdas)},
             "candidate_with_old_solver": D.choose_candidate(old_scores), "lambda_with_old_solver": old_lam,
             "candidate_recomputed_exact": new_cand, "lambda_recomputed_exact": new_lam,
             "candidate_stored": sel["candidate"], "lambda_stored": stored_lam,
             "old_scores": old_scores, "exact_scores": run_scores, "per_fit": per}
    return part1


def star(run: Path) -> dict:
    import torch

    from cvr.mdnet import PN_REL_VAR, load_md_model

    seeds = json.loads((run / "results.json").read_text())["selection"]["seeds"]
    part2 = {}
    for s in seeds:
        path = next((run / "models").glob(f"N-star_s{s}_e*.json")).with_suffix("")
        pn = load_md_model(path).net.pn
        ids = pn.global_dims.clone()
        g_mu, g_var = pn.global_stats()
        with torch.no_grad():
            var_d = pn._var(pn.dom_mean, pn.dom_msq)
            updated = pn.dom_updates > 0
            would_own = (var_d >= PN_REL_VAR * g_var) & ids & updated.unsqueeze(1)
        part2[f"N-star_s{s}"] = {"domains_with_statistics": int(updated.sum()), "id_dims": int(ids.sum()),
                                 "pairs_checked": int(updated.sum() * ids.sum()),
                                 "pairs_that_would_use_own_statistics_under_the_old_rule": int(would_own.sum()),
                                 "max_ratio_domain_var_to_global_var_on_id_dims":
                                     float((var_d[updated][:, ids] / g_var[ids]).max())}
    return part2


if __name__ == "__main__":
    part, run = sys.argv[1], Path(sys.argv[2])
    out = calib(run) if part == "calib" else star(run)
    (run / f"diagnostics_{part}.json").write_text(json.dumps({"run": str(run), part: out}, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_fit"}, indent=2))
