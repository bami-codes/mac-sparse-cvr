"""Study E Q2, per group (description after the official test, no verdict; docs/results/e_final_test.md).

    PYTHONPATH=. uv run python scripts/q2_per_group.py

For each G1 group: LogLoss after (iii) minus after (ii), the mean over the three seeds of the train-17 models, on V2
(train-19, where lambda was chosen) and on the official test (already evaluated once for Q2; this reads the same
predictions and calibrators and adds no claim). The mean over the 979 groups is the Q2 macro difference and must equal
the recorded V2 and test values. Groups are also split by their training positives (the G1 rule is < 123).
Output: runs/e_study/q2_per_group.json.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from cvr import metrics as M
from cvr import study_e_eval as Q
from cvr.checks import CheckFailed

OUT = Path("runs/e_study/q2_per_group.json")
BINS = ((0, 40), (40, 80), (80, 123))


def per_seed_diff(rid: str, name: str, y: np.ndarray, g: np.ndarray, m: np.ndarray, s: int) -> pl.DataFrame:
    p = Q.predictions(rid, name, y)["p"]
    q2, q3 = Q.load_calibrated(rid, "iii", "main", p[m], g[m])
    a = M.per_group(y[m], q2, g[m]).select("g", "n", "O", pl.col("logloss").alias("ll2"))
    b = M.per_group(y[m], q3, g[m]).select("g", pl.col("logloss").alias("ll3"))
    return a.join(b, on="g").select("g", "n", "O", (pl.col("ll3") - pl.col("ll2")).alias(f"d{s}"))


def describe(t: pl.DataFrame) -> dict:
    d = t["d"].to_numpy()
    zero = t.filter(pl.col("O") == 0)
    seeds = t.select([f"d{s}" for s in Q.SEEDS]).to_numpy()
    bins = []
    for lo, hi in BINS:
        bd = t.filter((pl.col("pos") >= lo) & (pl.col("pos") < hi))["d"].to_numpy()
        bins.append(
            {
                "train_positives": f"[{lo},{hi})",
                "groups": int(bd.size),
                "mean_diff": float(bd.mean()),
                "better": int((bd < 0).sum()),
                "worse": int((bd > 0).sum()),
            }
        )
    return {
        "groups": int(d.size),
        "macro_diff": float(d.mean()),
        "better": int((d < 0).sum()),
        "worse": int((d > 0).sum()),
        "better_in_all_3_seeds": int((seeds < 0).all(axis=1).sum()),
        "worse_in_all_3_seeds": int((seeds > 0).all(axis=1).sum()),
        "sum_of_improvements": float(-d[d < 0].sum()),
        "sum_of_worsenings": float(d[d > 0].sum()),
        "groups_without_positives": zero.height,
        "without_positives_mean_diff": float(zero["d"].mean()),
        "without_positives_better": int((zero["d"] < 0).sum()),
        "by_training_positives": bins,
    }


def main() -> dict:
    codes = Q.token_codes()
    g1 = Q.group_sets(codes)["G1"]
    choice = json.loads(Path("runs/e_study/config_choice.json").read_text())
    stats = pl.read_parquet("runs/e_study/train_stats_ad_feat1.parquet").select(
        pl.col("ad_feat1").replace_strict(codes, return_dtype=pl.Int64).alias("g"), "clicks", "pos"
    )
    out = {}
    for name in (Q.V2, Q.TEST):
        g, y = Q.file_columns(name, codes)
        m = np.isin(g, g1)
        for model in ("MoAE", "BASE"):
            parts = [
                per_seed_diff(Q.run_id(model, choice[model]["lr"], choice[model]["lamda"], s), name, y, g, m, s)
                for s in Q.SEEDS
            ]
            t = parts[0]
            for d in parts[1:]:
                t = t.join(d.drop("n", "O"), on="g")
            t = t.with_columns(pl.mean_horizontal([f"d{s}" for s in Q.SEEDS]).alias("d")).join(stats, on="g")
            out[f"{name} {model}"] = describe(t)
    v2 = json.loads(Path("runs/e_study/q2/selection.json").read_text())["v2_bootstrap"]
    test = json.loads(Path("runs/e_study/frozen/result_q2.json").read_text())["secondary"]
    for model in ("MoAE", "BASE"):
        for name, rec in ((Q.V2, v2), (Q.TEST, test)):
            want = rec[f"{model}: iii - ii"]["diff"]
            if abs(out[f"{name} {model}"]["macro_diff"] - want) > 1e-12:
                raise CheckFailed(f"{model} {name}: the per-group mean differs from the recorded difference {want}")
    OUT.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
