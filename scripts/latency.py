"""Online/offline consistency check and single-thread latency for the saved models of one run.

For each saved base model (M1, first-seed M2 / M3 variants) with the run's selected calibration:
1) consistency: the serving path (cvr/serving.py, pure python/numpy) and the evaluation path
   (cvr/features.py + saved artifacts) must give the same features and calibrated predictions
   on the same raw rows;
2) latency: raw row dict -> mapping -> history features -> model -> calibration, in-process,
   single thread; batch=1 (1,000 requests) and batch=100 (100 requests). Network, feature-store
   lookups and parsing of the request are excluded.

Each base model is measured in its own subprocess (--base), because PyTorch and LightGBM cannot
share one process on this machine (see cvr/m3_worker.py).

Usage:
    PYTHONPATH=. uv run python scripts/latency.py runs/<name>/<stamp> data/raw/mac/train/train-19.parquet
"""

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from cvr.features import RAW_COLUMNS, featurize, user_columns
from cvr.m3data import encode_m3
from cvr.schema import GROUP_KEY, NUM_INPUTS, VOCAB_COLS
from cvr.serving import Server, selection

N_ROWS = 3000


def load_vocab_frames(folder: Path) -> dict[str, pl.DataFrame]:
    return {p.stem: pl.read_parquet(p) for p in folder.glob("*.parquet")}


def offline_predict(server: Server, run_dir: Path, raw: pl.DataFrame, feats: pl.DataFrame) -> np.ndarray:
    if server.base.startswith("M3"):
        p = server.model.predict(encode_m3(raw, feats, load_vocab_frames(run_dir / "id_vocab")))
    else:
        p = server.model.predict(feats)
    g = feats[GROUP_KEY].to_numpy()
    if server.method == "ii":
        return server.glob.predict(p)
    if server.method == "iv":
        n = np.array([server.group_clicks.get(int(x), 0) for x in g], dtype=np.float64)
        return server.fa.predict(p, n)
    return server.gs.predict(p, g)


def pct(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {"p50_ms": float(np.percentile(a, 50)), "p99_ms": float(np.percentile(a, 99)), "mean_ms": float(a.mean())}


def measure(run_dir: Path, v2_file: str, base: str, method: str) -> dict:
    table = pq.ParquetFile(v2_file).read_row_group(0, columns=RAW_COLUMNS).slice(0, N_ROWS)
    rows = table.to_pylist()
    raw = user_columns(pl.from_arrow(table))
    feats = featurize(raw, load_vocab_frames(run_dir / "vocab"))
    server = Server(run_dir, base, method)
    codes, num = server.features(rows)
    feat_diff = {"codes_mismatch": int(sum((codes[c] != feats[c].to_numpy()).sum() for c in VOCAB_COLS)),
                 "num_max_abs_diff": float(np.abs(num - feats.select(NUM_INPUTS).to_numpy()).max())}
    online = server.predict(rows)
    offline = offline_predict(server, run_dir, raw, feats)
    for _ in range(50):  # warm-up
        server.predict(rows[:1])
    b1 = []
    for i in range(1000):
        t = time.perf_counter()
        server.predict(rows[i:i + 1])
        b1.append(1000 * (time.perf_counter() - t))
    b100 = []
    for i in range(100):
        chunk = rows[(i * 100) % (N_ROWS - 100):][:100]
        t = time.perf_counter()
        server.predict(chunk)
        b100.append(1000 * (time.perf_counter() - t))
    return {"feature_consistency": feat_diff,
            "prediction_max_abs_diff_online_vs_offline": float(np.abs(online - offline).max()),
            "batch1": pct(b1), "batch100": pct(b100)}


def main(run: str, v2_file: str) -> None:
    run_dir = Path(run)
    sel = selection(run_dir)
    bases = sorted(p.stem for p in (run_dir / "models").glob("M*.json"))
    report = {"run": str(run_dir), "calibration_method": sel["calibration_method"], "rows": N_ROWS,
              "env": {"cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                            text=True).stdout.strip(),
                      "python": platform.python_version(), "torch_threads": 1, "lightgbm_threads": 1,
                      "process": "one subprocess per base model"},
              "models": {}}
    for base in bases:
        out = run_dir / f"latency_{base}.json"
        subprocess.run([sys.executable, __file__, run, v2_file, "--base", base, "--out", str(out)], check=True)
        report["models"][base] = json.loads(out.read_text())
        out.unlink()
        print(base, json.dumps(report["models"][base]))
    (run_dir / "latency.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {run_dir / 'latency.json'}")


if __name__ == "__main__":
    if "--base" in sys.argv:
        run_arg, v2_arg = sys.argv[1], sys.argv[2]
        base_arg = sys.argv[sys.argv.index("--base") + 1]
        out_arg = Path(sys.argv[sys.argv.index("--out") + 1])
        sel_ = selection(Path(run_arg))
        out_arg.write_text(json.dumps(measure(Path(run_arg), v2_arg, base_arg, sel_["calibration_method"])))
    else:
        main(sys.argv[1], sys.argv[2])
