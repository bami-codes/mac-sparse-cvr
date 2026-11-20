"""Compare GPU (pod) and Mac CPU predictions of the same checkpoint on the same rows (study E device check).

    python scripts/device_compare.py <gpu .npz> <cpu .npz> <out .json>

Passes only when the rows (users, labels) are the same and every probability column differs by at most TOL, with
every difference finite. max() skips NaN, so without that check a NaN column would pass.
"""

import json
import sys
from pathlib import Path

import numpy as np

TOL = 1e-5


def compare(gpu: dict, cpu: dict) -> dict:
    gcols, ccols = {k for k in gpu if k.startswith("p_")}, {k for k in cpu if k.startswith("p_")}
    cols = sorted(gcols & ccols)  # a column missing on one side is a failure, recorded below, not a crash
    same_rows = (
        gcols == ccols and np.array_equal(gpu["user"], cpu["user"]) and np.array_equal(gpu["y_last"], cpu["y_last"])
    )
    diff = {k: float(np.abs(np.asarray(gpu[k], np.float64) - np.asarray(cpu[k], np.float64)).max()) for k in cols}
    finite = {k: bool(np.isfinite(gpu[k]).all() and np.isfinite(cpu[k]).all()) for k in cols}
    ok = bool(same_rows and cols and all(finite[k] and np.isfinite(diff[k]) and diff[k] <= TOL for k in cols))
    return {
        "rows": int(np.asarray(gpu["p_last"]).size),
        "same_rows_and_labels": bool(same_rows),
        "max_abs_diff": diff,
        "all_finite": finite,
        "tolerance": TOL,
        "passed": ok,
    }


if __name__ == "__main__":
    g, c, out = sys.argv[1:4]
    rec = {"gpu": g, "cpu": c} | compare(dict(np.load(g)), dict(np.load(c)))
    Path(out).write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec))
    sys.exit(0 if rec["passed"] else 1)
