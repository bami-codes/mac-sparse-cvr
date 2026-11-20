"""Study C final evaluation: score every saved network on prepared arrays, in a PyTorch-only process.

    python -m cvr.final_eval_c_nn <run> <arrays dir> <out dir>

Writes <out dir>/<family>_s<seed>.npy for N-int, N-shared, N-head, N-star and N-star-noPN with the same prediction
batch as the development run (cvr/study_c_worker.PREDICT_BATCH), so the V2 dry run can require exact reproduction.
"""

import sys
import tomllib
from pathlib import Path

import numpy as np
import torch

from .checks import require_finite_probs
from .domain_eval import SEEDS
from .mddata import load_md
from .mdnet import load_md_model
from .serving_c import model_path
from .study_c_worker import PREDICT_BATCH

FAMILIES = ["N-int", "N-shared", "N-head", "N-star", "N-star-noPN"]


def main(run: str, arrays: str, out: str) -> None:
    # same thread count as the training worker that recorded the V2 predictions (see cvr/final_eval_m3.py)
    torch.set_num_threads(tomllib.loads((Path(run) / "config.toml").read_text())["models"]["nn"]["threads"])
    a = load_md(Path(arrays), mmap=True)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fam in FAMILIES:
        for s in SEEDS:
            p = load_md_model(model_path(Path(run), fam, s)).predict(a, batch=PREDICT_BATCH)
            require_finite_probs(f"{fam}_s{s}", p)
            np.save(out_dir / f"{fam}_s{s}.npy", p)


if __name__ == "__main__":
    main(*sys.argv[1:4])
