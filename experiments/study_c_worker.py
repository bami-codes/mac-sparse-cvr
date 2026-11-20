"""Study C networks in a separate PyTorch-only process (OpenMP: PyTorch and LightGBM cannot share a process here).

Job folder (written by experiments/study_c.py):
    job.json      {"cat_sizes", "dom_col", "dim", "batch", "threads", "models_dir", "reload_rows", "data_dir",
                   "runs": [{"name", "variant", "seed", "lr", "epochs", "eval_epochs", "save"}]}
    <data_dir>/train/ v1/ v2/  MDArrays (experiments/mddata.save_md); <data_dir>/y_train.npy
Outputs:
    preds/<name>_e<epoch>_<role>.npy   predictions for every evaluated checkpoint
    result.json                        training info per checkpoint and same-batch reload checks

The two phases of §16.4 are two jobs: "tune" (every variant x learning rate, seed 1, checkpoints after epochs 1
and 2, nothing saved) and "final" (the chosen learning rate and epochs, every seed, models saved and reloaded).
Calibration and selection happen in the calling process, which never imports torch.

    python -m experiments.study_c_worker runs/<...>/nn_tune
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from cvr.checks import RELOAD_TOL, require_close, require_finite_probs

from .mddata import load_md
from .mdnet import MDModel, load_md_model, save_md_model

PREDICT_BATCH = 16384


def same_batch_rows(wanted: int, n: int, batch: int = PREDICT_BATCH) -> int:
    """Rows for the reload check that keep every prediction batch identical to the full-split pass (a shorter last
    batch would change float32 rounding and is not a same-batch comparison)."""
    if n <= batch:
        return n
    return min(n, max(batch, (wanted // batch) * batch))


def main(job_dir: str) -> None:
    job_path = Path(job_dir)
    job = json.loads((job_path / "job.json").read_text())
    data_dir = Path(job.get("data_dir", job_path))  # arrays shared by the tune and final jobs of one run
    train = load_md(data_dir / "train", mmap=False)
    evals = {r: load_md(data_dir / r, mmap=True) for r in ("v1", "v2")}
    y = np.load(data_dir / "y_train.npy")
    (job_path / "preds").mkdir(exist_ok=True)
    models_dir = Path(job["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"fitted": {}, "reload_check": {}}
    for run in job["runs"]:
        t = time.time()
        checkpoints = MDModel.fit(
            train,
            y,
            job["cat_sizes"],
            job["dom_col"],
            run["variant"],
            run["seed"],
            dim=job["dim"],
            epochs=run["epochs"],
            batch=job["batch"],
            lr=run["lr"],
            threads=job["threads"],
        )
        fit_seconds = time.time() - t
        for m in checkpoints:
            e = m.info["epochs"]
            if e not in run["eval_epochs"]:
                continue
            name = f"{run['name']}_e{e}"
            preds = {r: m.predict(a, batch=PREDICT_BATCH) for r, a in evals.items()}
            for r, p in preds.items():
                require_finite_probs(f"{name} {r}", p)
                np.save(job_path / "preds" / f"{name}_{r}.npy", p)
            result["fitted"][name] = m.info | {"fit_seconds_all_epochs": fit_seconds}
            if run["save"]:
                save_md_model(m, job["cat_sizes"], job["dom_col"], models_dir / name)
                k = same_batch_rows(job["reload_rows"], evals["v2"].n)
                p_re = load_md_model(models_dir / name).predict(evals["v2"].take(np.arange(k)), batch=PREDICT_BATCH)
                require_finite_probs(f"reloaded {name}", p_re)
                result["reload_check"][name] = require_close(  # same batching -> same-batch tolerance
                    f"reload {name}", float(np.abs(p_re - preds["v2"][:k]).max()), RELOAD_TOL
                )
        print(
            f"{run['name']} ({run['variant']}, lr {run['lr']}, {run['epochs']} epochs) done in {time.time() - t:.0f}s",
            flush=True,
        )
    (job_path / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
