"""One categorical mapping built from the TRAINING files and reused for every split.

Codes: 0 = value not in the training files (UNKNOWN), 1 = PAD token (MISSING),
2.. = training values ordered by (count desc, value asc), so the mapping is deterministic
and independent of file order or thread scheduling.
"""

import hashlib
import json
from pathlib import Path

import polars as pl

from .schema import MISSING, PAD, UNKNOWN, VOCAB_COLS


def count_values(frames: list[pl.DataFrame], cols: list[str] = VOCAB_COLS,
                 counts: dict[str, pl.DataFrame] | None = None) -> dict[str, pl.DataFrame]:
    """Value counts per column; pass `counts` to add one more file without keeping earlier ones."""
    counts = dict(counts or {})
    for df in frames:
        for c in cols:
            vc = df.group_by(c).agg(pl.len().alias("n")).rename({c: "value"})
            counts[c] = vc if c not in counts else (
                pl.concat([counts[c], vc]).group_by("value").agg(pl.col("n").sum()))
    return counts


def build_vocab(counts: dict[str, pl.DataFrame], min_count: int = 1) -> dict[str, pl.DataFrame]:
    vocab = {}
    for c, vc in counts.items():
        known = (vc.filter((pl.col("value") != PAD) & (pl.col("n") >= min_count))
                 .sort(["n", "value"], descending=[True, False])
                 .with_row_index("code", offset=2)
                 .select(pl.col("value"), pl.col("code").cast(pl.Int32)))
        pad = pl.DataFrame({"value": [PAD], "code": [MISSING]}, schema={"value": pl.String, "code": pl.Int32})
        vocab[c] = pl.concat([pad, known])
    return vocab


def encode(df: pl.DataFrame, vocab: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Replace each vocabulary column with its integer code; unseen values -> UNKNOWN."""
    return df.with_columns([
        pl.col(c).replace_strict(v["value"], v["code"], default=UNKNOWN, return_dtype=pl.Int32).alias(c)
        for c, v in vocab.items() if c in df.columns
    ])


def fingerprint(vocab: dict[str, pl.DataFrame]) -> str:
    h = hashlib.sha256()
    for c in sorted(vocab):
        h.update(c.encode())
        h.update(vocab[c].sort("code").write_csv().encode())
    return h.hexdigest()[:16]


def save(vocab: dict[str, pl.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for c, v in vocab.items():
        v.write_parquet(out_dir / f"{c}.parquet")
    (out_dir / "sizes.json").write_text(json.dumps({c: v.height + 1 for c, v in vocab.items()}, indent=2))


def sizes(vocab: dict[str, pl.DataFrame]) -> dict[str, int]:
    """Number of distinct codes per column, including UNKNOWN (0) and MISSING (1)."""
    return {c: int(v["code"].max()) + 1 for c, v in vocab.items()}
