"""EDA E1, E2, E5 and MAC premise checks E6.1-E6.2 on one MAC training file.

Writes aggregate statistics only (no row-level data) to a JSON file.

Usage:
    uv run python scripts/eda_premise.py data/raw/mac/train/train-17.parquet runs/eda/train-17.json
"""

import json
import resource
import sys
import time
from pathlib import Path

import polars as pl

LABELS = ["last", "first", "linear", "mta"]
SCALAR = [
    "user_id", "user_feat0", "user_feat1", "user_feat2", "user_feat3", "user_feat4", "user_feat5",
    "context_feat0", "context_feat1", "context_feat2", "context_feat3",
    "ad_feat0", "ad_feat1", "ad_feat2", "ad_feat3", "ad_feat4", "ad_feat5",
]
SEQ = ["ad_feat0_seq", "ad_feat1_seq", "ad_feat2_seq"]
PAD = "dbee74ee5fbf"  # [추정] padding/missing hash token, see E5 diagnostics
MM = ["mm_feat0_seq", "mm_feat1_seq"]


def peak_rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, kilobytes on Linux
    return (r if sys.platform == "darwin" else r * 1024) / 1e9


def main(src: str, out: str) -> None:
    t0 = time.time()
    df = pl.read_parquet(src, columns=SCALAR + SEQ + MM + LABELS)
    report: dict = {"file": Path(src).name, "read_seconds": round(time.time() - t0, 1)}
    df = df.with_columns((pl.col("last") > 0).cast(pl.Int8).alias("y"))
    n = df.height

    # E1: sizes and label distributions
    report["E1"] = {
        "rows": n,
        "label_positive_rate": {c: df.select((pl.col(c) > 0).mean()).item() for c in LABELS},
        "label_n_unique": {c: df[c].n_unique() for c in LABELS},
        "label_max": {c: float(df[c].max()) for c in LABELS},
        "n_unique": {c: df[c].n_unique() for c in SCALAR},
        "null_count": {c: df[c].null_count() for c in SCALAR + SEQ + MM},
    }

    # E2: functional-dependency checks (support or contradict guesses; never prove meaning)
    def share_single(key: str, val: str) -> float:
        g = df.group_by(key).agg(pl.col(val).n_unique().alias("k"))
        return g.select((pl.col("k") == 1).mean()).item()

    pairs = [("ad_feat0", "ad_feat1"), ("ad_feat0", "ad_feat2"), ("ad_feat0", "ad_feat3"),
             ("ad_feat0", "ad_feat4"), ("ad_feat1", "ad_feat3"), ("ad_feat1", "ad_feat4"),
             ("ad_feat2", "ad_feat1"), ("ad_feat1", "ad_feat2")]
    report["E2_share_of_keys_with_single_value"] = {f"{a}->{b}": share_single(a, b) for a, b in pairs}

    # E5: sequence lengths and mm score summaries
    lens = df.select([pl.col(c).list.len().alias(c) for c in SEQ + MM])
    e5: dict = {
        "seq_len_quantiles": {f"{c}_q{q}": lens[c].quantile(q) for c in lens.columns for q in [0.0, 0.5, 1.0]},
        "seq_len_equal_across_cols_share": lens.select(
            (pl.col("ad_feat0_seq") == pl.col("ad_feat1_seq")).mean()).item(),
    }
    # padding diagnostics: a pad token would dominate positions and fill whole rows
    for c in SEQ:
        top = (df.select(pl.col(c).explode().value_counts(sort=True)).unnest(c).head(3))
        e5[f"{c}_top_tokens_share_of_positions"] = [
            {"token": r[c], "share": r["count"] / (n * 20)} for r in top.to_dicts()]
        e5[f"{c}_rows_single_unique_token_share"] = df.select(
            (pl.col(c).list.n_unique() == 1).mean()).item()
    for c in MM:
        flat = df.select(pl.col(c).explode())[c]
        e5[f"{c}_zero_share_of_positions"] = (flat == 0).mean()
        e5[f"{c}_quantiles"] = {q: flat.quantile(q) for q in [0.0, 0.1, 0.5, 0.9, 1.0]}
    report["E5"] = e5

    # E6.2: does purchase history carry signal? (counts of target value inside the history)
    idx = df.with_row_index("rid").select(["rid", "ad_feat0", "ad_feat1", "ad_feat2"] + SEQ)
    feats = []
    for tgt, seq in [("ad_feat0", "ad_feat0_seq"), ("ad_feat1", "ad_feat1_seq"), ("ad_feat2", "ad_feat2_seq")]:
        m = (idx.select(["rid", tgt, seq]).explode(seq)
             .group_by("rid").agg(((pl.col(seq) == pl.col(tgt)) & (pl.col(seq) != PAD) & (pl.col(tgt) != PAD))
                                  .sum().alias(f"match_{tgt}")))
        feats.append(m)
    report["E6_2_target_is_pad_share"] = {t: df.select((pl.col(t) == PAD).mean()).item()
                                          for t in ["ad_feat0", "ad_feat1", "ad_feat2"]}
    report["E5_pad_in_scalar_columns_share"] = {c: df.select((pl.col(c) == PAD).mean()).item()
                                                for c in SCALAR if df[c].dtype == pl.String}
    mm = df.with_row_index("rid").select(
        ["rid"] + [pl.col(c).list.max().alias(f"{c}_max") for c in MM])
    enriched = df.with_row_index("rid").select(["rid", "y", "ad_feat1"])
    for m in feats + [mm]:
        enriched = enriched.join(m, on="rid", how="left")
    enriched = enriched.with_columns(pl.col("^match_.*$").fill_null(0))

    e62 = {}
    for c in ["match_ad_feat0", "match_ad_feat1", "match_ad_feat2"]:
        e62[c] = {
            "share_rows_gt0": enriched.select((pl.col(c) > 0).mean()).item(),
            "cvr_if_gt0": enriched.filter(pl.col(c) > 0)["y"].mean(),
            "cvr_if_0": enriched.filter(pl.col(c) == 0)["y"].mean(),
        }
    for c in [f"{x}_max" for x in MM]:
        q = enriched.select(pl.col(c).qcut(5, labels=[f"q{i}" for i in range(1, 6)],
                                           allow_duplicates=True).alias("bin"), "y")
        e62[c] = {"cvr_by_quintile": q.group_by("bin").agg(pl.col("y").mean()).sort("bin").to_dicts(),
                  "null_share": enriched[c].null_count() / n}
    report["E6_2_history_signal"] = e62

    # E6.1: sparse but evaluable groups (one file only; multi-file counts will be larger)
    # missing-token group is reported separately and excluded from sparse-group candidates
    miss = df.filter(pl.col("ad_feat1") == PAD)
    report["E6_1_missing_group"] = {"clicks": miss.height, "click_share": miss.height / n,
                                    "positives_by_label": {c: miss.select((pl.col(c) > 0).sum()).item()
                                                           for c in LABELS}}
    g = (df.filter(pl.col("ad_feat1") != PAD)
         .group_by("ad_feat1").agg(pl.len().alias("clicks"), pl.col("y").sum().alias("pos")))
    grid = {}
    for n_min in [200, 500, 1000, 2000]:
        for c_min in [10, 20, 50]:
            sel = g.filter((pl.col("clicks") >= n_min) & (pl.col("pos") < c_min))
            grid[f"n_min={n_min},c_min={c_min}"] = {
                "groups": sel.height, "click_share": sel["clicks"].sum() / n}
    report["E6_1_ad_feat1_groups"] = {
        "n_groups": g.height,
        "clicks_quantiles": {q: g["clicks"].quantile(q) for q in [0.5, 0.9, 0.99]},
        "pos_quantiles": {q: g["pos"].quantile(q) for q in [0.5, 0.9, 0.99]},
        "groups_with_ge200_clicks": g.filter(pl.col("clicks") >= 200).height,
        "sparse_grid": grid,
    }

    report["total_seconds"] = round(time.time() - t0, 1)
    report["peak_rss_gb"] = round(peak_rss_gb(), 2)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    print(json.dumps(report, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
