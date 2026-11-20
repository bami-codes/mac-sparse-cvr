"""Turn one raw MAC file into a compact numeric frame with the shared vocabulary.

Output columns:
- `user_key` (int64 from the 12-hex user id; stable across library versions), `bucket` (0-99)
- vocabulary columns as integer codes, `is_missing` (ad_feat1 is the PAD token)
- history features (NUM_INPUTS): counts of the target's item / ad_feat1 / ad_feat2 value inside
  the purchase sequence (PAD excluded), whether the history is empty, and mm score max/mean
- `y` = (last > 0) and `label_first/linear/mta` (auxiliary labels; never model inputs)
"""

from pathlib import Path

import polars as pl

from cvr.schema import AUX_LABELS, MM_COLS, PAD, RAW_LABELS, SEQ_PAIRS, TARGET, VOCAB_COLS

from .vocab import encode

RAW_COLUMNS = ["user_id", "ad_feat0"] + VOCAB_COLS + [s for _, s in SEQ_PAIRS] + MM_COLS + RAW_LABELS


def user_columns(df: pl.DataFrame) -> pl.DataFrame:
    bad = df.select((~pl.col("user_id").str.contains(r"^[0-9a-f]{12}$")).sum()).item()
    if bad:
        raise ValueError(f"{bad} user_id values are not 12-hex strings")
    key = pl.col("user_id").str.to_integer(base=16, dtype=pl.Int64)
    return df.with_columns(key.alias("user_key"), (key % 100).cast(pl.Int16).alias("bucket"))


def read_raw(
    path: Path, buckets: tuple[int, int] | None = None, user_pct: int | None = None, columns: list[str] = RAW_COLUMNS
) -> pl.DataFrame:
    """Read the needed raw columns; optionally keep only some user buckets (smoke runs)."""
    df = user_columns(pl.read_parquet(path, columns=sorted(set(columns) | {"user_id"})))
    if buckets is not None:
        df = df.filter(pl.col("bucket").is_between(buckets[0], buckets[1] - 1))
    if user_pct is not None:  # independent digit of the key, so it does not interact with buckets
        df = df.filter((pl.col("user_key") // 100) % 100 < user_pct)
    return df


def history_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_row_index("rid")
    out = df.select("rid")
    for tgt, seq in SEQ_PAIRS:
        m = (
            df.select(["rid", tgt, seq])
            .explode(seq)
            .group_by("rid")
            .agg(
                ((pl.col(seq) == pl.col(tgt)) & (pl.col(seq) != PAD) & (pl.col(tgt) != PAD))
                .sum()
                .cast(pl.Int16)
                .alias(f"match_{tgt}")
            )
        )
        out = out.join(m, on="rid", how="left")
    out = out.join(
        df.select(
            "rid",
            ((pl.col("ad_feat0_seq").list.n_unique() == 1) & (pl.col("ad_feat0_seq").list.first() == PAD))
            .cast(pl.Int8)
            .alias("hist_empty"),
            *[
                e
                for c in MM_COLS
                for e in (
                    pl.col(c).list.max().cast(pl.Float32).alias(c.replace("_seq", "_max")),
                    pl.col(c).list.mean().cast(pl.Float32).alias(c.replace("_seq", "_mean")),
                )
            ],
        ),
        on="rid",
        how="left",
    )
    return out.sort("rid").drop("rid")


def featurize(raw: pl.DataFrame, vocab: dict[str, pl.DataFrame]) -> pl.DataFrame:
    hist = history_features(raw)
    base = raw.select(
        "user_key",
        "bucket",
        (pl.col("ad_feat1") == PAD).cast(pl.Int8).alias("is_missing"),
        *VOCAB_COLS,
        (pl.col(TARGET) > 0).cast(pl.Int8).alias("y"),
        *[(pl.col(c) > 0).cast(pl.Int8).alias(f"label_{c}") for c in AUX_LABELS],
    )
    return encode(pl.concat([base, hist], how="horizontal"), vocab)
