"""Tell apart differences between machines from differences inside one machine, on the serving check rows.

    PYTHONPATH=. uv run python scripts/serving_device_diff.py dump NAME       # on each machine
    PYTHONPATH=. uv run python scripts/serving_device_diff.py compare mac linux

dump serves the 2,999 check rows one at a time and as a single batch. compare lines up machine vs machine next to
one row vs a batch on the same machine, and both vs the saved evaluation predictions (local CPU, batches of 4,096).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import serving_bench as S

OUT = S.DIR


def dump(name: str) -> dict:
    meta, plan, rows = S.load_inputs()
    srv = S.servers(meta)
    check_rows = [rows[int(i)] for i in plan["check"]]
    arrays = {}
    for m, server in srv.items():
        single = np.array([server.raw([r])[0] for r in check_rows])
        batch = server.raw(check_rows)
        arrays |= {
            f"{m}_single": single,
            f"{m}_batch": batch,
            f"{m}_saved": plan[f"p_{m}"],
            f"{m}_cal_single": server.calibrate(single, check_rows),
        }
    np.savez(OUT / f"raw_{name}.npz", **arrays)
    return {"name": name, "machine": S.machine(), "code": S.code_version()}


def stats(d: np.ndarray, p: np.ndarray) -> dict:
    a = np.abs(d)
    return {
        "max": float(a.max()),
        "p99": float(np.percentile(a, 99)),
        "median": float(np.median(a)),
        "max_relative": float((a / np.maximum(p, 1e-12)).max()),
        "rows_above_1e-6": int((a > 1e-6).sum()),
    }


def compare(a: str, b: str) -> dict:
    x, y = dict(np.load(OUT / f"raw_{a}.npz")), dict(np.load(OUT / f"raw_{b}.npz"))
    out = {}
    for m in S.L.MODELS:
        saved = x[f"{m}_saved"]
        if not np.array_equal(saved, y[f"{m}_saved"]):
            raise ValueError("the two dumps hold different saved predictions")
        p = saved
        out[m] = {
            f"{a} single vs saved": stats(x[f"{m}_single"] - saved, p),
            f"{b} single vs saved": stats(y[f"{m}_single"] - saved, p),
            f"{a} single vs {a} batch": stats(x[f"{m}_single"] - x[f"{m}_batch"], p),
            f"{b} single vs {b} batch": stats(y[f"{m}_single"] - y[f"{m}_batch"], p),
            f"{a} batch vs saved": stats(x[f"{m}_batch"] - saved, p),
            f"{b} batch vs saved": stats(y[f"{m}_batch"] - saved, p),
            f"{a} single vs {b} single": stats(x[f"{m}_single"] - y[f"{m}_single"], p),
            f"{a} vs {b} calibrated (iii), single": stats(x[f"{m}_cal_single"] - y[f"{m}_cal_single"], p),
            "probability range": [float(p.min()), float(np.median(p)), float(p.max())],
        }
    (OUT / f"device_diff_{a}_{b}.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1:]
    if len(cmd) == 2 and cmd[0] == "dump":
        print(json.dumps(dump(cmd[1]), indent=1))
    elif len(cmd) == 3 and cmd[0] == "compare":
        print(json.dumps(compare(cmd[1], cmd[2]), indent=1))
    else:
        raise SystemExit(__doc__)
