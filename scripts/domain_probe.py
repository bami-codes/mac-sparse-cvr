"""Discovery probe for study C (docs/DESIGN.md §16): is `context_feat1` a usable domain axis?

Run only on files already used for development (train-17, train-19). It reads one context column and the
labels, and reports per-value traffic and conversion, plus how the values cut across categories and users.

    uv run python scripts/domain_probe.py data/raw/mac/train/train-17.parquet data/raw/mac/train/train-19.parquet \
        runs/eda/domain_probe.json
"""

import json
import sys
from pathlib import Path

import polars as pl

DOMAIN = "context_feat1"
SMALL_SHARE = 0.01  # the share cut looked at during discovery; the study fixes its own rule on training files


def per_value(path: str) -> tuple[pl.DataFrame, dict]:
    df = pl.read_parquet(path, columns=["user_id", DOMAIN, "ad_feat1", "last"])
    df = df.with_columns((pl.col("last") > 0).cast(pl.Int64).alias("y"))
    n = df.height
    g = (
        df.group_by(DOMAIN)
        .agg(
            pl.len().alias("clicks"),
            pl.col("y").sum().alias("pos"),
            pl.col("user_id").n_unique().alias("users"),
            pl.col("ad_feat1").n_unique().alias("categories"),
        )
        .with_columns((pl.col("clicks") / n).alias("share"), (pl.col("pos") / pl.col("clicks")).alias("cvr"))
        .sort([pl.col("clicks"), pl.col(DOMAIN)], descending=[True, False])
    )
    small = g.filter(pl.col("share") < SMALL_SHARE)
    large = g.filter(pl.col("share") >= SMALL_SHARE)
    cat_spread = df.group_by("ad_feat1").agg(pl.col(DOMAIN).n_unique().alias("k"))["k"]
    user_spread = df.group_by("user_id").agg(pl.col(DOMAIN).n_unique().alias("k"))["k"]
    s = {
        "file": Path(path).name,
        "rows": n,
        "cvr": float(df["y"].mean()),
        "n_values": g.height,
        "top_share": float(g["share"][0]),
        "cvr_range_all": [float(g["cvr"].min()), float(g["cvr"].max())],
        "cvr_range_share_ge_1pct": [float(large["cvr"].min()), float(large["cvr"].max())],
        "small": {
            "n": small.height,
            "click_share": float(small["share"].sum()),
            "positives": int(small["pos"].sum()),
            "positives_median": float(small["pos"].median()),
            "n_lt10_positives": int((small["pos"] < 10).sum()),
            "min_clicks": int(small["clicks"].min()),
            "n_lt1000_clicks": int((small["clicks"] < 1000).sum()),
        },
        "categories_per_value_median": float(g["categories"].median()),
        "values_per_category_quantiles": {str(q): float(cat_spread.quantile(q)) for q in (0.5, 0.9, 0.99)},
        "share_users_in_gt1_values": float((user_spread > 1).mean()),
    }
    return g, s


def main(paths: list[str], out: str) -> None:
    tables, summaries = {}, []
    for p in paths:
        g, s = per_value(p)
        tables[s["file"]] = g
        summaries.append(s)
    a, b = list(tables)[:2]
    j = (
        tables[a]
        .select(DOMAIN, pl.col("share").alias("share_a"), pl.col("cvr").alias("cvr_a"))
        .join(
            tables[b].select(DOMAIN, pl.col("share").alias("share_b"), pl.col("cvr").alias("cvr_b")),
            on=DOMAIN,
            how="full",
            coalesce=True,
        )
    )
    small_a = set(tables[a].filter(pl.col("share") < SMALL_SHARE)[DOMAIN].to_list())
    small_b = set(tables[b].filter(pl.col("share") < SMALL_SHARE)[DOMAIN].to_list())
    stability = {
        "files": [a, b],
        "values_only_in_one": int(j.null_count().select("share_a", "share_b").sum_horizontal()[0]),
        "corr_share": float(j.drop_nulls().select(pl.corr("share_a", "share_b")).item()),
        "corr_cvr": float(j.drop_nulls().select(pl.corr("cvr_a", "cvr_b")).item()),
        "same_small_set": small_a == small_b,
    }
    res = {"summaries": summaries, "stability": stability, "per_value": {k: v.to_dicts() for k, v in tables.items()}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({"summaries": summaries, "stability": stability}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:-1], sys.argv[-1])
