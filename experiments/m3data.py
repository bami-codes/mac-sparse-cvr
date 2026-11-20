"""M3 input arrays (no PyTorch here, so the LightGBM process never loads a second OpenMP runtime).

- a separate id vocabulary for ad_feat0/1/2 counted over training targets and purchase sequences
  (M1/M2 codes are untouched); PAD -> MISSING (masked in the network), rare/unseen -> UNKNOWN
- arrays: other categorical codes, target ids, 20-step sequence ids, per-step visual-similarity
  scores, and the same summary history features as M1/M2
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from cvr.schema import CAT_INPUTS, MISSING, MM_COLS, NUM_INPUTS, SEQ_PAIRS, UNKNOWN, assert_no_label_inputs

SEQ_LEN = 20
ID_COLS = [t for t, _ in SEQ_PAIRS]  # ad_feat0 (item), ad_feat1, ad_feat2
OTHER_CATS = [c for c in CAT_INPUTS if c not in ID_COLS]
assert_no_label_inputs(OTHER_CATS + NUM_INPUTS)


# ---------------------------------------------------------------- id vocabulary and arrays
def count_ids(raw: pl.DataFrame, counts: dict | None = None) -> dict[str, pl.DataFrame]:
    """Counts of each id value over the target column and its purchase-sequence column."""
    counts = dict(counts or {})
    for tgt, seq in SEQ_PAIRS:
        both = pl.concat([raw.select(pl.col(tgt).alias("v")), raw.select(pl.col(seq).explode().alias("v"))])
        vc = both.group_by("v").agg(pl.len().alias("n")).rename({"v": "value"})
        counts[tgt] = (
            vc if tgt not in counts else (pl.concat([counts[tgt], vc]).group_by("value").agg(pl.col("n").sum()))
        )
    return counts


@dataclass
class M3Arrays:
    other: np.ndarray  # (n, len(OTHER_CATS)) int32
    target: np.ndarray  # (n, 3) int32
    seq: np.ndarray  # (n, 20, 3) int32
    mm: np.ndarray  # (n, 20, 2) float32
    num: np.ndarray  # (n, len(NUM_INPUTS)) float32 (raw, standardized at train time)
    coverage: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.other.shape[0]


def encode_m3(raw: pl.DataFrame, feats: pl.DataFrame, id_vocab: dict[str, pl.DataFrame]) -> M3Arrays:
    """`raw` is the raw frame of one file; `feats` is its featurize() output (same row order)."""
    n = raw.height
    lens = raw.select(
        [pl.col(s).list.len().alias(s) for _, s in SEQ_PAIRS] + [pl.col(m).list.len().alias(m) for m in MM_COLS]
    )
    if not (lens.to_numpy() == SEQ_LEN).all():
        raise ValueError("expected every purchase sequence to have length 20")
    tgt_cols, seq_cols, coverage = [], [], {}
    for tgt, seq in SEQ_PAIRS:
        v = id_vocab[tgt]
        enc = lambda e, v=v: e.replace_strict(v["value"], v["code"], default=UNKNOWN, return_dtype=pl.Int32)  # noqa: E731
        tgt_cols.append(raw.select(enc(pl.col(tgt)))[tgt].to_numpy())
        s = raw.select(enc(pl.col(seq).explode()))[seq].to_numpy().reshape(n, SEQ_LEN)
        seq_cols.append(s)
        real = s != MISSING
        coverage[tgt] = {
            "target_unknown_share": float((tgt_cols[-1] == UNKNOWN).mean()),
            "seq_unknown_share_of_nonpad": float((s[real] == UNKNOWN).mean()) if real.any() else None,
        }
    mm = np.stack([raw.select(pl.col(m).explode())[m].to_numpy().reshape(n, SEQ_LEN) for m in MM_COLS], axis=-1)
    return M3Arrays(
        other=feats.select(OTHER_CATS).to_numpy().astype(np.int32),
        target=np.stack(tgt_cols, axis=1).astype(np.int32),
        seq=np.stack(seq_cols, axis=-1).astype(np.int32),
        mm=mm.astype(np.float32),
        num=feats.select(NUM_INPUTS).to_numpy().astype(np.float32),
        coverage=coverage,
    )


def concat_arrays(parts: list[M3Arrays]) -> M3Arrays:
    return M3Arrays(
        *(np.concatenate([getattr(p, f) for p in parts]) for f in ("other", "target", "seq", "mm", "num")),
        coverage={"per_file": [p.coverage for p in parts]},
    )


# ---------------------------------------------------------------- network


FIELDS = ("other", "target", "seq", "mm", "num")


def save_arrays(a: M3Arrays, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for f in FIELDS:
        np.save(folder / f"{f}.npy", getattr(a, f))


def load_arrays(folder: Path, mmap: bool = True) -> M3Arrays:
    mode = "r" if mmap else None
    return M3Arrays(*(np.load(folder / f"{f}.npy", mmap_mode=mode) for f in FIELDS))
