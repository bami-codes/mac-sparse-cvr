"""Error rate of the agent's tests on synthetic data with a KNOWN answer (docs/DESIGN.md §14.2).

The agent's model comparisons are judged by a paired user bootstrap with the fitted models held fixed.
Noise models cannot tell whether that test has its nominal error rate (a noise model is a different fitted
model, so it can really differ). This script runs the menu's own tools (`Menu.from_frame`) on data where
the true difference is known, with clicks clustered within users and rare positives as in MAC.

Data-generating process (one simulation; fresh users each time):
- clicks per user: drawn from the empirical clicks-per-user distribution of the development menu
  (runs/agent_menu, train-10/11; only user ids are read, no labels), users drawn until the target rows;
- truth: p = sigmoid(a + x + u), x ~ N(0, 1) per click, u ~ N(0, 0.7^2) per user, a set so that the mean
  CVR is 0.0156 (MAC discovery rate). The 20-simulation smoke run used a = logit(0.0156), which gives a
  mean CVR near 3% (the spread raises the mean); the intercept was recalibrated before the full run;
- two "fitted models" with equal quality by construction: sigmoid(a + x + c z) and sigmoid(a + x - c z),
  z ~ N(0, 1) per user, c = 0.3. Swapping z -> -z swaps the models, so the mean LogLoss difference is
  exactly 0 (null). The errors are shared within a user, so differences are correlated within users;
- a worse model sigmoid(a + 0.6 x + c z): known non-zero difference (alternative), its true value
  computed from the exact Bernoulli expectation over 4M draws;
- an association feature independent of everything (per-user rate q ~ Beta(1, 7/3), mean 0.3) and a
  per-click segment flag with probability 0.5 (interaction null; the same users fall in both arms).

Checked per scale (target rows): null rejection rate at alpha = 0.05 for ablation_effect, interaction and
association; for the alternative, 95% CI coverage of the true difference and power.

Decision rule (fixed before this script was first run): a tool passes at a scale if, among simulations
where it was computable (arm minimums met), the null rejection rate is <= 0.065 and, for the ablation
alternative, the CI coverage is >= 0.935 (2,000 simulations: the standard error at 0.05 is about 0.005).
If a tool fails at a scale above the current minimum (MIN_POSITIVES = 10 per arm), the minimum for the final
run is raised to the median smallest-arm positives of the smallest scale from which every tool passes at
that scale and all larger ones. MIN_POSITIVES is one constant for every arm, so where the tools' medians
differ at that scale the largest is used (conservative reading, written down after the run: the rule did
not say which tool's median; `decide`).

Two gates (minimum per compared arm), stored with the results:
- `diagnostic` (rows >= 200, positives >= 10, no user floor): the setting the decision rule was applied to;
  docs/results/agent_bootstrap_check.{md,json}.
- `operating` (the menu's MIN_ROWS / MIN_POSITIVES / MIN_USERS, i.e. after the decision): error rates of the
  rule actually used, among simulations that pass it; no decision is made from it;
  docs/results/agent_bootstrap_check_operating.{md,json}.

    uv run python scripts/agent_bootstrap_check.py [--gate diagnostic|operating] [--sims 2000]
    uv run python scripts/agent_bootstrap_check.py --rewrite     # outputs again from the saved JSON
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import brentq
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvr.agent.experiments import (FEATURE_GROUPS, MIN_POSITIVES, MIN_ROWS, MIN_USERS,  # noqa: E402
                                   NOISE_REPLICATES, Menu)
from cvr.metrics import bce  # noqa: E402

SD_U, C, WEAK_X, CVR = 0.7, 0.3, 0.6, 0.0156


def _intercept() -> float:
    """a with E[sigmoid(a + e)] = CVR for e ~ N(0, 1 + SD_U^2) (Gauss-Hermite quadrature)."""
    t, w = np.polynomial.hermite_e.hermegauss(80)
    sd = np.sqrt(1 + SD_U ** 2)
    return float(brentq(lambda a: (w * expit(a + sd * t)).sum() / w.sum() - CVR, -10, 0))


A = _intercept()
SCALES = [700, 2_000, 7_000, 20_000, 100_000]
ALPHA, MAX_REJECT, MIN_COVER = 0.05, 0.065, 0.935
SEED = 20261118
GATES = {"diagnostic": {"min_rows": 200, "min_positives": 10, "min_users": 1},
         "operating": {"min_rows": MIN_ROWS, "min_positives": MIN_POSITIVES, "min_users": MIN_USERS}}


def expected_loss(p, q):
    """E_y[LogLoss] for y ~ Bernoulli(p) and prediction q (no label noise)."""
    q = np.clip(q, 1e-7, 1 - 1e-7)
    return -(p * np.log(q) + (1 - p) * np.log(1 - q))


def true_alternative(rng, n=4_000_000):
    x, u, z = rng.normal(size=n), rng.normal(0, SD_U, n), rng.normal(size=n)
    p = expit(A + x + u)
    d = expected_loss(p, expit(A + x + C * z)) - expected_loss(p, expit(A + WEAK_X * x + C * z))
    return float(d.mean()), float(d.std() / np.sqrt(n))


def simulate(rng, sizes, rows, gate: dict):
    k = rng.choice(sizes, size=int(rows / sizes.mean() * 2) + 20)
    k = k[: np.searchsorted(np.cumsum(k), rows) + 1]
    user = np.repeat(np.arange(k.size), k)
    n = user.size
    x = rng.normal(size=n)
    u, z = rng.normal(0, SD_U, k.size)[user], rng.normal(size=k.size)[user]
    q = rng.beta(1, 7 / 3, k.size)[user]
    y = (rng.uniform(size=n) < expit(A + x + u)).astype(np.int8)
    p_full = expit(A + x + C * z).astype(np.float32)
    cols = {"user_key": user, "y": y, "group_class": np.full(n, "dense"),
            "hist_empty": (rng.uniform(size=n) < 0.5).astype(np.int8),
            "match_ad_feat0": np.zeros(n, np.int16), "match_ad_feat2": np.zeros(n, np.int16),
            "match_ad_feat1": (rng.uniform(size=n) < q).astype(np.int16), "high_visual": np.zeros(n, bool),
            "p_full": p_full}
    for g in FEATURE_GROUPS:
        cols[f"p_no_{g}"] = p_full
    cols["p_no_history_match"] = expit(A + x - C * z).astype(np.float32)        # null pair
    cols["p_no_visual_similarity"] = expit(A + WEAK_X * x + C * z).astype(np.float32)  # alternative
    for r in range(NOISE_REPLICATES):
        cols[f"p_full_noise_{r}"] = p_full
    df = pl.DataFrame(cols)
    return Menu.from_frame(df, {"config": {"data": {"confirmation": ["synthetic"]}}}, **gate), y, user


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, float(c - h)), min(1.0, float(c + h))]


def decide(results: list[dict], current_min: int = 10) -> dict:
    """Apply the decision rule above to the per-scale results."""
    scales = sorted({r["rows"] for r in results})
    ok = {n: all(r.get("pass") for r in results if r["rows"] == n) for n in scales}
    first = next((n for i, n in enumerate(scales) if all(ok[m] for m in scales[i:])), None)
    if first is None:
        return {"all_pass_from_rows": None, "min_positives": None, "note": "no scale passes for every tool"}
    medians = {r["tool"]: r["median_smallest_arm_positives"] for r in results if r["rows"] == first}
    failed_above = [r for r in results if r.get("pass") is False and r["median_smallest_arm_positives"] > current_min]
    new_min = max(current_min, int(np.ceil(max(medians.values())))) if failed_above else current_min
    return {"all_pass_from_rows": first, "medians_at_that_scale": medians, "current_min_positives": current_min,
            "failures_above_current_min": len(failed_above), "min_positives": new_min}


def write(out: Path, meta: dict, results: list[dict]):
    meta.setdefault("gate_name", "diagnostic")  # results saved before the gate was a parameter used this one
    meta.setdefault("gate", GATES["diagnostic"])
    diagnostic = meta["gate_name"] == "diagnostic"
    d = decide(results) if diagnostic else None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"meta": meta, "results": results, "decision": d}, indent=2))
    title = ("Agent test error rates on synthetic data with a known answer" if diagnostic else
             "Agent test error rates under the operating gate (synthetic data, known answer)")
    lines = [f"# {title}", "",
             f"`scripts/agent_bootstrap_check.py --gate {meta['gate_name']}`, gate per compared arm {meta['gate']}, "
             f"seed {meta['seed']}, {meta['sims_per_scale']} simulations per scale, "
             f"alpha {meta['alpha']}. Pass: null rejection <= {meta['max_null_rejection']} and 95% CI coverage >= "
             f"{meta['min_coverage']} (decision rule in the script).",
             f"Clicks per user from the development menu ({meta['cluster_sizes']['users']:,} users, mean "
             f"{meta['cluster_sizes']['mean']:.2f}). True alternative difference {meta['sanity']['truth']:+.6f} "
             f"(Monte Carlo s.e. {meta['sanity']['truth_mc_se']:.1e}).", "",
             "| rows | tool | computable | median positives / users / converting users (smallest arm) | "
             "rejection rate [Wilson 95%] | CI coverage | pass |",
             "|---:|---|---:|---|---|---:|---|"]
    for r in results:
        size = f"{r['median_smallest_arm_positives']:.0f}"
        if "median_smallest_arm_users" in r:
            size += f" / {r['median_smallest_arm_users']:.0f} / {r['median_smallest_arm_converting_users']:.0f}"
        if r["computable"] == 0:
            lines.append(f"| {r['rows']:,} | {r['tool']} | 0 | {size} | - | - | - |")
            continue
        lo, hi = r["rejection_wilson95"]
        lines.append(f"| {r['rows']:,} | {r['tool']} | {r['computable']} | {size} | "
                     f"{r['rejection_rate']:.3f} [{lo:.3f}, {hi:.3f}] | {r['ci_coverage']:.3f} | {r['pass']} |")
    lines += ["", "For the alternative row, the rejection rate is the power. Medians are over all simulations, "
              "computable or not.", ""]
    if not diagnostic:
        lines += ["Error rates here are among simulations that pass the operating gate (computable column). "
                  "This describes the rule actually used on this synthetic configuration; it is not a general "
                  "sample-size guarantee, and no threshold is chosen from it."]
        out.with_suffix(".md").write_text("\n".join(lines) + "\n")
        return d
    lines += [
              f"**Decision**: every tool passes from {d['all_pass_from_rows']:,} rows on (medians of the smallest arm "
              f"there: {', '.join(f'{k} {v:.0f}' for k, v in d['medians_at_that_scale'].items())}). "
              f"{d['failures_above_current_min']} results failed above the old minimum of "
              f"{d['current_min_positives']} positives, so the minimum per compared arm becomes "
              f"**{d['min_positives']} positives** (largest median, one constant for every arm)."
              if d["all_pass_from_rows"] else f"**Decision**: {d['note']}"]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--menu", default="runs/agent_menu/confirmation_predictions.parquet")
    ap.add_argument("--gate", choices=list(GATES), default="diagnostic")
    ap.add_argument("--out", default=None, help="default: docs/results/agent_bootstrap_check[_operating]")
    ap.add_argument("--rewrite", action="store_true", help="write the outputs again from the saved JSON")
    a = ap.parse_args()
    a.out = a.out or ("docs/results/agent_bootstrap_check" + ("" if a.gate == "diagnostic" else "_operating"))
    gate = GATES[a.gate]
    if a.rewrite:
        saved = json.loads(Path(a.out).with_suffix(".json").read_text())
        print(write(Path(a.out), saved["meta"], saved["results"]))
        return
    # sorted: group_by returns groups in no fixed order, and rng.choice draws by POSITION, so the same seed drew
    # different users in each run (the first diagnostic run cannot be reproduced exactly for this reason)
    sizes = np.sort(pl.read_parquet(a.menu, columns=["user_key"]).group_by("user_key").len()["len"].to_numpy())
    rng = np.random.default_rng(SEED)
    truth, truth_se = true_alternative(rng)
    # check the expected-loss formula against realized labels once (same truth, larger sample)
    menu, y, _ = simulate(rng, sizes, 2_000_000, gate)
    d = bce(y, menu.df["p_full"].to_numpy()) - bce(y, menu.df["p_no_visual_similarity"].to_numpy())
    sanity = {"intercept": A, "truth": truth, "truth_mc_se": truth_se, "realized_mean_2m_rows": float(d.mean()),
              "realized_cvr_2m_rows": float(y.mean())}
    print(sanity, flush=True)

    tools = {
        "ablation_effect (null)": lambda m: m.ablation_effect("history_match", "all"),
        "interaction (null)": lambda m: m.interaction("history_match", "has_history", "no_history"),
        "association (null)": lambda m: m.association("same_category_in_history", "all"),
        "ablation_effect (alternative)": lambda m: m.ablation_effect("visual_similarity", "all"),
    }
    results = []
    for rows in SCALES:
        t0 = time.time()
        acc = {name: {"computable": 0, "reject": 0, "cover": 0, "smallest_arm_pos": [], "smallest_arm_users": [],
                      "smallest_arm_conv_users": []} for name in tools}
        for _ in range(a.sims):
            menu, y, _ = simulate(rng, sizes, rows, gate)
            seg = menu.df["hist_empty"].to_numpy() == 1
            feat = menu.df["match_ad_feat1"].to_numpy() > 0
            arm_pos = {"ablation_effect (null)": y.sum(), "ablation_effect (alternative)": y.sum(),
                       "interaction (null)": min(y[seg].sum(), y[~seg].sum()),
                       "association (null)": min(y[feat].sum(), y[~feat].sum())}
            for name, call in tools.items():
                r = call(menu)
                acc[name]["smallest_arm_pos"].append(int(arm_pos[name]))
                small = min(r["arms"].values(), key=lambda st: st["positives"])
                acc[name]["smallest_arm_users"].append(small["users"])
                acc[name]["smallest_arm_conv_users"].append(small["converting_users"])
                if r["status"] != "ok":
                    continue
                acc[name]["computable"] += 1
                acc[name]["reject"] += r["p_value"] < ALPHA
                lo, hi = r["ci95"]
                acc[name]["cover"] += lo <= (truth if "alternative" in name else 0.0) <= hi
        for name, v in acc.items():
            n = v["computable"]
            row = {"rows": rows, "tool": name, "sims": a.sims, "computable": n,
                   "median_smallest_arm_positives": float(np.median(v["smallest_arm_pos"])),
                   "median_smallest_arm_users": float(np.median(v["smallest_arm_users"])),
                   "median_smallest_arm_converting_users": float(np.median(v["smallest_arm_conv_users"])),
                   "rejection_rate": v["reject"] / n if n else None, "rejection_wilson95": wilson(v["reject"], n),
                   "ci_coverage": v["cover"] / n if n else None, "coverage_wilson95": wilson(v["cover"], n)}
            if n:
                row["pass"] = (row["ci_coverage"] >= MIN_COVER if "alternative" in name
                               else row["rejection_rate"] <= MAX_REJECT and row["ci_coverage"] >= MIN_COVER)
            results.append(row)
            print(json.dumps(row), flush=True)
        print(f"rows {rows}: {time.time() - t0:.0f}s", flush=True)

    meta = {"seed": SEED, "sims_per_scale": a.sims, "alpha": ALPHA, "max_null_rejection": MAX_REJECT,
            "min_coverage": MIN_COVER, "dgp": {"a": A, "sd_u": SD_U, "c": C, "weak_x": WEAK_X, "cvr": CVR},
            "cluster_sizes": {"users": int(sizes.size), "mean": float(sizes.mean()),
                              "q50_q90_q99_max": np.quantile(sizes, [0.5, 0.9, 0.99]).tolist() + [int(sizes.max())]},
            "sanity": sanity, "gate_name": a.gate, "gate": gate}
    print(write(Path(a.out), meta, results))


if __name__ == "__main__":
    main()
