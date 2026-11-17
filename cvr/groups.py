"""Group statistics from the TRAINING rows and group classes for any split.

Classes (by the training statistics of the row's ad_feat1 group):
- "missing": ad_feat1 is the PAD token (reported separately, never a sparse group)
- "unseen":  ad_feat1 value not in the training files
- "G1":      clicks >= n_min and positives < c_min  (the primary sparse groups)
- "small":   clicks < n_min (too few clicks to evaluate the group on its own)
- "dense":   positives >= c_dense
- "mid":     everything else
"""

import polars as pl

from .schema import GROUP_KEY, MISSING, UNKNOWN

CLASSES = ["G1", "small", "mid", "dense", "unseen", "missing"]


def train_stats(train: pl.DataFrame, key: str = GROUP_KEY) -> pl.DataFrame:
    return train.group_by(key).agg(pl.len().alias("tr_clicks"), pl.col("y").sum().alias("tr_pos"))


def classify(df: pl.DataFrame, stats: pl.DataFrame, n_min: int, c_min: int, c_dense: int,
             key: str = GROUP_KEY) -> pl.DataFrame:
    tc, tp = pl.col("tr_clicks"), pl.col("tr_pos")
    out = (df.select(key).join(stats, on=key, how="left", maintain_order="left")
           .with_columns(tc.fill_null(0), tp.fill_null(0)))
    cls = (pl.when(pl.col(key) == MISSING).then(pl.lit("missing"))
           .when((pl.col(key) == UNKNOWN) | (tc == 0)).then(pl.lit("unseen"))
           .when((tc >= n_min) & (tp < c_min)).then(pl.lit("G1"))
           .when(tc < n_min).then(pl.lit("small"))
           .when(tp >= c_dense).then(pl.lit("dense"))
           .otherwise(pl.lit("mid")))
    return out.with_columns(cls.alias("group_class")).select("tr_clicks", "tr_pos", "group_class")


def summary(stats: pl.DataFrame, n_min: int, c_min: int, c_dense: int) -> dict:
    s = stats.filter(pl.col(GROUP_KEY) != MISSING)
    total = stats["tr_clicks"].sum()
    g1 = s.filter((pl.col("tr_clicks") >= n_min) & (pl.col("tr_pos") < c_min))
    dense = s.filter(pl.col("tr_pos") >= c_dense)
    miss = stats.filter(pl.col(GROUP_KEY) == MISSING)
    return {
        "definition": {"key": GROUP_KEY, "n_min": n_min, "c_min": c_min, "c_dense": c_dense,
                       "selected_on": "training files only"},
        "train_groups_nonmissing": s.height,
        "G1_groups": g1.height, "G1_train_click_share": float(g1["tr_clicks"].sum() / total),
        "G1_train_positives": int(g1["tr_pos"].sum()),
        "dense_groups": dense.height,
        "missing_train_clicks": int(miss["tr_clicks"].sum()), "missing_train_positives": int(miss["tr_pos"].sum()),
    }
