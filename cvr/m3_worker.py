"""Train and predict the M3 variants in a SEPARATE process that uses PyTorch only.

Why a separate process: PyTorch and LightGBM ship separate OpenMP runtimes on this machine, and
using both in one process segfaults or hangs (checked with a small script: torch first -> exit 139,
LightGBM first -> hang; KMP_DUPLICATE_LIB_OK does not help). The pipeline never imports torch.

Job folder (written by cvr/pipeline.py):
    job.json             variants, seeds, hyper-parameters, embedding table sizes, models_dir, reload_rows
    train/ v1/ v2/       M3 arrays as .npy (cvr/m3data.save_arrays)
    y_train.npy
Outputs:
    preds/<name>_<role>.npy   predictions for V1 and V2
    result.json               per-model training info and reload checks

    python -m cvr.m3_worker runs/<...>/m3_job
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from .checks import RELOAD_TOL, require_close
from .m3data import M3Arrays, load_arrays
from .nn import M3Model, load_m3, save_m3


def main(job_dir: str) -> None:
    job_path = Path(job_dir)
    job = json.loads((job_path / "job.json").read_text())
    train = load_arrays(job_path / "train", mmap=False)  # random access during training -> keep in RAM
    evals = {r: load_arrays(job_path / r, mmap=True) for r in ("v1", "v2")}
    y = np.load(job_path / "y_train.npy")
    (job_path / "preds").mkdir(exist_ok=True)
    models_dir = Path(job["models_dir"])
    result: dict = {"fitted": {}, "reload_check": {}}
    for variant in job["variants"]:
        for seed in job["seeds"]:
            t = time.time()
            m = M3Model.fit(train, y, job["other_sizes"], job["id_sizes"], variant, seed, dim=job["dim"],
                            epochs=job["epochs"], batch=job["batch"], lr=job["lr"], device=job["device"],
                            threads=job["threads"])
            name = f"M3_{variant}_s{seed}"
            preds = {r: m.predict(a) for r, a in evals.items()}
            for r, p in preds.items():
                np.save(job_path / "preds" / f"{name}_{r}.npy", p)
            result["fitted"][name] = m.info | {"seconds": time.time() - t}
            # every seed is saved so the final test can reuse exactly these fits
            save_m3(m, job["other_sizes"], job["id_sizes"], models_dir / name)
            k = min(job["reload_rows"], evals["v2"].n)
            head = M3Arrays(*(np.asarray(getattr(evals["v2"], f)[:k]) for f in ("other", "target", "seq", "mm", "num")))
            result["reload_check"][name] = require_close(
                f"reload {name}", float(np.abs(load_m3(models_dir / name).predict(head) - preds["v2"][:k]).max()),
                RELOAD_TOL)
            print(f"{name} done in {time.time() - t:.0f}s", flush=True)
    (job_path / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
