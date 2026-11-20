"""Fix the ad group lists from the training files alone, before any model result exists.

    uv run python -m cvr.groups      # G1, the comparison set C and small context groups, with their SHA-256

Groups are kept as the raw MAC token (hashed string), so they do not depend on any vocabulary. The PAD token is the
missing value and belongs to no group list. A click is positive when last > 0 (the official label).
"""

import hashlib
import json
import sys
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from .schema import PAD

DATA = Path("data/raw/mac/train")
TRAIN = [f"train-{i:02d}.parquet" for i in range(18)]  # what the research models were trained on
OUT = Path("runs/eval")
BASE_ROWS_G1, BASE_ROWS_C = 10_883_174, 10_499_793  # training rows of the first study and of study C
G1_BASE = {"n_min": 500, "c_min": 20, "c_dense": 50}
SMALL_SHARE, SMALL_MIN_CLICKS_BASE = 0.01, 3000
FIRST_STUDY_RUN = Path("runs/dev_train15_17_m3/20261117T110713Z")


def sha_of(values) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def train_stats(files: list[Path], key: str) -> pl.DataFrame:
    """Clicks and positives per raw value of `key`, summed over the files (columns read: key and last)."""
    parts = []
    for f in files:
        t = pq.read_table(f, columns=[key, "last"])
        parts.append(
            pl.from_arrow(t).group_by(key).agg(pl.len().alias("clicks"), (pl.col("last") > 0).sum().alias("pos"))
        )
    return (
        pl.concat(parts)
        .group_by(key)
        .agg(pl.col("clicks").sum(), pl.col("pos").sum())
        .with_columns(pl.col("clicks").cast(pl.Int64), pl.col("pos").cast(pl.Int64))
        .sort(key)
    )


def first_study_g1() -> set[str] | None:
    """The first study's 1,083 G1 groups as raw tokens (its frozen statistics and vocabulary), if the run is here."""
    stats, vocab = FIRST_STUDY_RUN / "frozen_group_stats.parquet", FIRST_STUDY_RUN / "vocab" / "ad_feat1.parquet"
    if not (stats.exists() and vocab.exists()):
        return None
    s = pl.read_parquet(stats)
    v = pl.read_parquet(vocab)
    g1 = s.filter((pl.col("tr_clicks") >= G1_BASE["n_min"]) & (pl.col("tr_pos") < G1_BASE["c_min"]))
    j = g1.join(v.rename({"code": "ad_feat1"}), on="ad_feat1", how="left")
    return set(j["value"].drop_nulls().to_list()) - {PAD}


def groups() -> dict:
    files = [DATA / f for f in TRAIN]
    rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    scale_g1, scale_c = rows / BASE_ROWS_G1, rows / BASE_ROWS_C
    n_min, c_min, c_dense = (round(v * scale_g1) for v in (G1_BASE["n_min"], G1_BASE["c_min"], G1_BASE["c_dense"]))
    small_min = round(SMALL_MIN_CLICKS_BASE * scale_c)

    ad = train_stats(files, "ad_feat1")
    if int(ad["clicks"].sum()) != rows:
        raise RuntimeError("ad_feat1 statistics do not cover every training row")
    known = ad.filter(pl.col("ad_feat1") != PAD)
    g1 = known.filter((pl.col("clicks") >= n_min) & (pl.col("pos") < c_min))
    comp = known.filter((pl.col("clicks") >= n_min) & (pl.col("pos") >= c_min))
    g1_abs = known.filter((pl.col("clicks") >= G1_BASE["n_min"]) & (pl.col("pos") < G1_BASE["c_min"]))
    ctx = train_stats(files, "context_feat1").with_columns((pl.col("clicks") / rows).alias("share"))
    small = ctx.filter(
        (pl.col("share") < SMALL_SHARE) & (pl.col("clicks") >= small_min) & (pl.col("context_feat1") != PAD)
    )

    first = first_study_g1()
    lists = {
        "G1": g1["ad_feat1"].to_list(),
        "C": comp["ad_feat1"].to_list(),
        "G1_abs500_20": g1_abs["ad_feat1"].to_list(),
        "small_context": small["context_feat1"].to_list(),
    }
    rec = {
        "training_files": TRAIN,
        "training_rows": rows,
        "positive_rule": "last > 0",
        "missing_token": PAD,
        "definitions": {
            "G1": f"ad_feat1 with training clicks >= {n_min} and positives < {c_min} (first study's 500 / 20 scaled by "
            f"{scale_g1:.3f} = {rows} / {BASE_ROWS_G1} training rows)",
            "C": f"ad_feat1 with training clicks >= {n_min} and positives >= {c_min} (same clicks condition, not G1)",
            "G1_abs500_20": "sensitivity: the first study's thresholds unscaled (clicks >= 500, positives < 20)",
            "small_context": f"context_feat1 with training share < {SMALL_SHARE:.0%} and clicks >= {small_min} (study C's "
            f"3,000 scaled by {scale_c:.3f})",
        },
        "thresholds": {"n_min": n_min, "c_min": c_min, "c_dense": c_dense, "small_min_clicks": small_min},
        "counts": {k: len(v) for k, v in lists.items()},
        "train_click_share": {
            "G1": float(g1["clicks"].sum() / rows),
            "C": float(comp["clicks"].sum() / rows),
            "G1_abs500_20": float(g1_abs["clicks"].sum() / rows),
            "small_context": float(small["clicks"].sum() / rows),
            "missing_ad_feat1": float(ad.filter(pl.col("ad_feat1") == PAD)["clicks"].sum() / rows),
        },
        "train_positives": {
            "G1": int(g1["pos"].sum()),
            "C": int(comp["pos"].sum()),
            "small_context": int(small["pos"].sum()),
        },
        "overlap_with_first_study_G1": None
        if first is None
        else {
            "first_study_groups": len(first),
            "in_new_G1": len(first & set(lists["G1"])),
            "in_new_C": len(first & set(lists["C"])),
            "below_new_click_minimum": len(first - set(lists["G1"]) - set(lists["C"])),
        },
        "sha256": {k: sha_of(v) for k, v in lists.items()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "groups.json"
    if dest.exists():
        raise RuntimeError(f"{dest} exists: the group lists are fixed once")
    ad.write_parquet(OUT / "train_stats_ad_feat1.parquet")
    ctx.write_parquet(OUT / "train_stats_context_feat1.parquet")
    (OUT / "group_lists.json").write_text(json.dumps(lists))
    dest.write_text(
        json.dumps(
            rec | {"group_lists_sha256": hashlib.sha256((OUT / "group_lists.json").read_bytes()).hexdigest()}, indent=2
        )
    )
    return rec


if __name__ == "__main__":
    if not sys.argv[1:]:
        print(json.dumps({k: v for k, v in groups().items() if k != "training_files"}, indent=2))
    else:
        raise SystemExit(__doc__)
