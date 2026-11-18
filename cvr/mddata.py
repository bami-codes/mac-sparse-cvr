"""Study C input arrays (no PyTorch here, so the LightGBM process never loads a second OpenMP runtime).

The networks of docs/DESIGN.md §16.4 use exactly the M2 inputs: the CAT_INPUTS codes of the shared vocabulary
and the NUM_INPUTS history summaries. The domain is the DOMAIN_KEY column of `cat`.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .schema import CAT_INPUTS, DOMAIN_KEY, NUM_INPUTS, assert_no_label_inputs

assert_no_label_inputs(CAT_INPUTS + NUM_INPUTS)
DOM_COL = CAT_INPUTS.index(DOMAIN_KEY)
FIELDS = ("cat", "num")


@dataclass
class MDArrays:
    cat: np.ndarray  # (n, len(CAT_INPUTS)) int32 vocabulary codes
    num: np.ndarray  # (n, len(NUM_INPUTS)) float32, raw (standardized with training statistics at fit time)

    @property
    def n(self) -> int:
        return self.cat.shape[0]

    def take(self, idx: np.ndarray) -> "MDArrays":
        return MDArrays(np.asarray(self.cat[idx]), np.asarray(self.num[idx]))


def encode_md(feats: pl.DataFrame) -> MDArrays:
    """`feats` is a featurize() output (cvr/features.py)."""
    return MDArrays(feats.select(CAT_INPUTS).to_numpy().astype(np.int32),
                    feats.select(NUM_INPUTS).to_numpy().astype(np.float32))


def concat_md(parts: list[MDArrays]) -> MDArrays:
    return MDArrays(*(np.concatenate([getattr(p, f) for p in parts]) for f in FIELDS))


def save_md(a: MDArrays, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for f in FIELDS:
        np.save(folder / f"{f}.npy", getattr(a, f))


def load_md(folder: Path, mmap: bool = True) -> MDArrays:
    mode = "r" if mmap else None
    return MDArrays(*(np.load(folder / f"{f}.npy", mmap_mode=mode) for f in FIELDS))
