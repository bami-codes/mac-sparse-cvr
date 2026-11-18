"""Study C: raw-row serving check against an independent reference, then single-thread latency per seed.

For every family (G-int, G-shared, N-int, N-shared, N-head, N-star) and every seed of a development run:
1) consistency (hard check, cvr/checks.serving_consistency): V2 rows are sampled from EVERY domain of the run's V2
   file (so small domains are present). The served probability of each row, computed ONE ROW AT A TIME by
   cvr/serving_c.py from the raw strings, is compared with a reference that does not use the serving code: the
   run's SAVED V2 base prediction of that row, with the saved (iii-d) calibrator applied to the row's domain code
   from the evaluation arrays (nn_data). Codes must match exactly, numeric features within NUM_TOL, probabilities
   within PRED_TOL (GBDT, float64) or NN_SINGLE_VS_BATCH_TOL (networks: float32 single row vs saved batch pass);
2) latency: raw row dict -> codes -> history features -> model -> calibration, in-process, single thread;
   batch=1 (1,000 requests after 50 warm-ups) and batch=100 (100 requests). Network, feature-store lookups and
   request parsing are excluded.

Each family runs in its own subprocess (PyTorch and LightGBM cannot share a process here); the parent imports
neither.

    PYTHONPATH=. uv run python scripts/latency_c.py runs/c_dev_train08_10/<run>
"""

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

FAMILIES = ["G-int", "G-shared", "N-int", "N-shared", "N-head", "N-star"]
PER_DOMAIN, RANDOM_ROWS, SAMPLE_SEED = 40, 2000, 20261118
RAW = ["user_id", "ad_feat0", "user_feat0", "user_feat1", "user_feat2", "user_feat3", "user_feat4", "user_feat5",
       "context_feat1", "context_feat2", "context_feat3", "ad_feat1", "ad_feat2", "ad_feat3", "ad_feat4", "ad_feat5",
       "ad_feat0_seq", "ad_feat1_seq", "ad_feat2_seq", "mm_feat0_seq", "mm_feat1_seq"]


def sample_indices(dom: np.ndarray) -> np.ndarray:
    """Up to PER_DOMAIN rows of every domain plus RANDOM_ROWS uniform rows (fixed seed), sorted, unique."""
    rng = np.random.default_rng(SAMPLE_SEED)
    parts = [rng.permutation(np.flatnonzero(dom == d))[:PER_DOMAIN] for d in np.unique(dom)]
    parts.append(rng.choice(dom.size, size=min(RANDOM_ROWS, dom.size), replace=False))
    return np.unique(np.concatenate(parts))


def read_rows(path: str, idx: np.ndarray) -> list[dict]:
    """Raw rows at file positions `idx` (sorted), reading only the row groups that hold them."""
    f = pq.ParquetFile(path)
    starts = np.r_[0, np.cumsum([f.metadata.row_group(i).num_rows for i in range(f.num_row_groups)])]
    rows: list[dict] = []
    for g in np.unique(np.searchsorted(starts, idx, side="right") - 1):
        local = idx[(idx >= starts[g]) & (idx < starts[g + 1])] - starts[g]
        rows += f.read_row_group(int(g), columns=RAW).take(local).to_pylist()
    return rows


def v2_rows(meta: dict, idx: np.ndarray) -> list[dict]:
    """Raw V2 rows at positions `idx` of the run's V2 arrays. A whole-file V2 is read by row group; a smoke V2
    (user buckets / user share of one file) is read with the run's own filter (cvr.features.read_raw), because
    then array positions are not file positions."""
    entries = meta["data"]["v2"]
    if len(entries) != 1:
        raise ValueError("this script expects one V2 file")
    e = entries[0]
    if e["buckets"] is None and meta["smoke_user_pct"] is None:
        return read_rows(e["file"], idx)
    import polars as pl

    from cvr.features import read_raw

    raw = read_raw(Path(e["file"]), tuple(e["buckets"]) if e["buckets"] else None, meta["smoke_user_pct"], columns=RAW)
    return raw.select(RAW)[pl.Series(idx)].to_dicts()


def base_pred_v2(run: Path, family: str, seed: int) -> np.ndarray:
    p = run / "preds" / f"{family}_s{seed}_v2.npy"
    if p.exists():
        return np.load(p, mmap_mode="r")
    hits = sorted((run / "nn_final" / "preds").glob(f"{family}_s{seed}_e*_v2.npy"))
    if len(hits) != 1:
        raise FileNotFoundError(f"saved V2 predictions of {family}_s{seed}: {hits}")
    return np.load(hits[0], mmap_mode="r")


def pct(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {"p50_ms": float(np.percentile(a, 50)), "p99_ms": float(np.percentile(a, 99)), "mean_ms": float(a.mean())}


def measure_family(run: Path, family: str) -> dict:
    from cvr.checks import NN_SINGLE_VS_BATCH_TOL, PRED_TOL, serving_consistency
    from cvr.mddata import DOM_COL, load_md
    from cvr.schema import CAT_INPUTS, NUM_INPUTS
    from cvr.serving_c import GBDT_FAMILIES, ServerC

    res = json.loads((run / "results.json").read_text())
    v2 = load_md(run / "nn_data" / "v2")
    if v2.n != res["meta"]["rows"]["v2"]:
        raise ValueError("nn_data/v2 rows differ from the run's V2 rows")
    dom = np.asarray(v2.cat[:, DOM_COL])
    idx = sample_indices(dom)
    rows = v2_rows(res["meta"], idx)
    tol = PRED_TOL if family in GBDT_FAMILIES else NN_SINGLE_VS_BATCH_TOL
    out = {"rows_checked": int(idx.size), "domains_checked": int(np.unique(dom[idx]).size), "pred_tol": tol, "seeds": {}}
    for seed in res["selection"]["seeds"]:
        server = ServerC(run, family, seed)
        codes, num = server.features(rows)
        online = np.array([server.predict([r])[0] for r in rows])  # the single-request path that is served
        ref = server.gs.predict(np.asarray(base_pred_v2(run, family, seed))[idx], dom[idx])  # independent reference
        consistency = serving_consistency({c: codes[c] for c in CAT_INPUTS},
                                          {c: np.asarray(v2.cat[idx, j]) for j, c in enumerate(CAT_INPUTS)},
                                          num, np.asarray(v2.num[idx], dtype=np.float64), online, ref, pred_tol=tol)
        for i in range(50):  # warm-up
            server.predict(rows[i % len(rows):][:1])
        b1 = []
        for i in range(1000):
            r = rows[i % len(rows):][:1]
            t = time.perf_counter()
            server.predict(r)
            b1.append(1000 * (time.perf_counter() - t))
        b100 = []
        for i in range(100):
            chunk = rows[(i * 100) % max(1, len(rows) - 100):][:100]
            t = time.perf_counter()
            server.predict(chunk)
            b100.append(1000 * (time.perf_counter() - t))
        out["seeds"][str(seed)] = {"server": server.info, "consistency": consistency, "batch1": pct(b1),
                                   "batch100": pct(b100)}
    out["p99_ms_by_seed"] = {s: v["batch1"]["p99_ms"] for s, v in out["seeds"].items()}
    return out


def main(run_arg: str) -> None:
    run = Path(run_arg)
    report = {"run": str(run),
              "env": {"cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                            text=True).stdout.strip(),
                      "python": platform.python_version(), "threads": 1, "process": "one subprocess per family"},
              "sampling": {"per_domain": PER_DOMAIN, "random_rows": RANDOM_ROWS, "seed": SAMPLE_SEED},
              "families": {}}
    for fam in FAMILIES:
        tmp = run / f"latency_c_{fam}.json"
        subprocess.run([sys.executable, __file__, run_arg, "--family", fam, "--out", str(tmp)], check=True)
        report["families"][fam] = json.loads(tmp.read_text())
        tmp.unlink()
        f = report["families"][fam]
        print(fam, {"p99_ms_by_seed": f["p99_ms_by_seed"],
                    "max_pred_diff": max(v["consistency"]["prediction_max_abs_diff"] for v in f["seeds"].values())},
              flush=True)
    (run / "latency_c.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {run / 'latency_c.json'}")


if __name__ == "__main__":
    if "--family" in sys.argv:
        fam_arg = sys.argv[sys.argv.index("--family") + 1]
        out_arg = Path(sys.argv[sys.argv.index("--out") + 1])
        out_arg.write_text(json.dumps(measure_family(Path(sys.argv[1]), fam_arg)))
    else:
        main(sys.argv[1])
