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


def classify(
    df: pl.DataFrame,
    stats: pl.DataFrame,
    n_min: int,
    c_min: int,
    c_dense: int,
    p0: float | None = None,
    key: str = GROUP_KEY,
) -> pl.DataFrame:
    """`group_class` is the primary definition (observed training positives).

    If `p0` (training positive rate) is given, `group_class_exp` adds the sensitivity definition
    that replaces observed positives with EXPECTED positives tr_clicks * p0, so the class does
    not condition on each group's observed labels: G1exp = n_min <= tr_clicks and tr_clicks * p0 < c_min.
    """
    tc, tp = pl.col("tr_clicks"), pl.col("tr_pos")
    out = (
        df.select(key)
        .join(stats, on=key, how="left", maintain_order="left")
        .with_columns(tc.fill_null(0), tp.fill_null(0))
    )
    special = (
        pl.when(pl.col(key) == MISSING)
        .then(pl.lit("missing"))
        .when((pl.col(key) == UNKNOWN) | (tc == 0))
        .then(pl.lit("unseen"))
    )
    cls = (
        special.when((tc >= n_min) & (tp < c_min))
        .then(pl.lit("G1"))
        .when(tc < n_min)
        .then(pl.lit("small"))
        .when(tp >= c_dense)
        .then(pl.lit("dense"))
        .otherwise(pl.lit("mid"))
    )
    cols = [cls.alias("group_class")]
    if p0 is not None:
        cols.append(
            special.when((tc >= n_min) & (tc * p0 < c_min))
            .then(pl.lit("G1exp"))
            .otherwise(pl.lit("other"))
            .alias("group_class_exp")
        )
    keep = ["tr_clicks", "tr_pos", "group_class"] + (["group_class_exp"] if p0 is not None else [])
    return out.with_columns(cols).select(keep)


def summary(stats: pl.DataFrame, n_min: int, c_min: int, c_dense: int, p0: float | None = None) -> dict:
    s = stats.filter(pl.col(GROUP_KEY) != MISSING)
    total = stats["tr_clicks"].sum()
    g1 = s.filter((pl.col("tr_clicks") >= n_min) & (pl.col("tr_pos") < c_min))
    dense = s.filter(pl.col("tr_pos") >= c_dense)
    miss = stats.filter(pl.col(GROUP_KEY) == MISSING)
    exp = {}
    if p0 is not None:
        g1e = s.filter((pl.col("tr_clicks") >= n_min) & (pl.col("tr_clicks") * p0 < c_min))
        both = g1.join(g1e.select(GROUP_KEY), on=GROUP_KEY, how="inner")
        exp = {
            "p0": p0,
            "G1exp_click_range": [n_min, c_min / p0],
            "G1exp_groups": g1e.height,
            "G1exp_train_click_share": float(g1e["tr_clicks"].sum() / total),
            "G1_and_G1exp_overlap": both.height,
        }
    return {
        "definition": {
            "key": GROUP_KEY,
            "n_min": n_min,
            "c_min": c_min,
            "c_dense": c_dense,
            "selected_on": "training files only",
        },
        "train_groups_nonmissing": s.height,
        "G1_groups": g1.height,
        "G1_train_click_share": float(g1["tr_clicks"].sum() / total),
        "G1_train_positives": int(g1["tr_pos"].sum()),
        "sensitivity_expected_positives": exp,
        "dense_groups": dense.height,
        "missing_train_clicks": int(miss["tr_clicks"].sum()),
        "missing_train_positives": int(miss["tr_pos"].sum()),
    }
