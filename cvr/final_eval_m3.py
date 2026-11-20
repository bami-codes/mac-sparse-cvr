"""Predict every saved M3 model on one evaluation file, in a PyTorch-only process (never imports LightGBM).

python -m cvr.final_eval_m3 <run_dir> <eval_file> <out_dir> [<bucket_lo> <bucket_hi> <user_pct>]
"""

import sys
import tomllib
from pathlib import Path

import numpy as np
import polars as pl
import torch

from .features import featurize, read_raw
from .m3data import encode_m3
from .nn import load_m3


def main(
    run_dir: str, eval_file: str, out_dir: str, lo: str | None = None, hi: str | None = None, pct: str | None = None
) -> None:
    run, out = Path(run_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Predict with the thread count the training worker used for the recorded V2 predictions. On x86 another count
    # changes the float32 rounding and the dry run's 1e-9 comparison fails (Apple CPUs give the same bits either way).
    torch.set_num_threads(tomllib.loads((run / "config.toml").read_text())["models"]["m3"]["threads"])
    vocab = {p.stem: pl.read_parquet(p) for p in (run / "vocab").glob("*.parquet")}
    id_vocab = {p.stem: pl.read_parquet(p) for p in (run / "id_vocab").glob("*.parquet")}
    buckets = (int(lo), int(hi)) if lo is not None else None
    raw = read_raw(Path(eval_file), buckets, int(pct) if pct not in (None, "None") else None)
    arrays = encode_m3(raw, featurize(raw, vocab), id_vocab)
    del raw
    for spec in sorted((run / "models").glob("M3_*.json")):
        np.save(out / f"{spec.stem}.npy", load_m3(spec.with_suffix("")).predict(arrays))
        print(f"{spec.stem} predicted", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
