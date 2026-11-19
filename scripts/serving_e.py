"""Study E serving check and latency on this Mac (docs/DESIGN.md §18.8), for BASE and MoAE (train-17, seed 2026).

    PYTHONPATH=. uv run python scripts/serving_e.py

1. Consistency (hard check): about 3,000 V2 rows (uniform, plus sparse-group rows), each served ONE row at a time
   from its raw columns. The raw probability must equal the saved evaluation prediction of that row, and the
   calibrated (iii) probability must equal the frozen calibrator applied to the saved prediction (loaded separately
   from the server), within 1e-6 (float32 networks, one row vs a batch).
2. Latency, one thread, in process: one click (50 warm-ups + 5,000 requests in a fixed uniform order of V2 rows), and a
   synthetic request of 100 ad candidates for one user (20 warm-ups + 300 requests). A synthetic request takes the
   ad and context features and the two precomputed similarity sequences of 100 real V2 rows and gives all of them
   one user's features and purchase history; the similarity values are read as given (MAC precomputes them), so
   computing them, network, parsing and feature-store lookups are not timed. The time covers the raw values ->
   vocabulary codes -> tensors -> model -> calibration.
3. Rounds: this Mac's other processes (an editor using three to four cores) moved single-run p99 by up to 4x, so the
   two models are measured alternately in 10 rounds (1,000 clicks and 100 requests each, different requests per round)
   and the median p50/p99 over rounds is reported with its range. Study C's acceptance rule is recorded per round
   (no other python of this repository, at most 400% CPU used by other processes at the start and end).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from cvr import pymal_run as R
from cvr import study_e_eval as Q
from cvr.checks import NN_SINGLE_VS_BATCH_TOL, CheckFailed
from cvr.serving_e import ServerE
from cvr.sysinfo import cpu_name
from scripts.latency_c import accepted, environment

SEED, N_UNIFORM, N_SPARSE = 20261120, 2000, 1000
WARMUP, REQUESTS, REQ_WARMUP, REQ_N, CANDIDATES = 50, 5000, 20, 300, 100
ROUNDS, ROUND_CLICKS, ROUND_REQUESTS = 10, 1000, 100  # alternate the models; report the median over rounds
OUT = Path("runs/e_study/serving.json")


def read_rows(path: Path, idx: np.ndarray, cols: list[str]) -> list[dict]:
    f = pq.ParquetFile(path)
    starts = np.r_[0, np.cumsum([f.metadata.row_group(i).num_rows for i in range(f.num_row_groups)])]
    rows = []
    for gi in np.unique(np.searchsorted(starts, idx, side="right") - 1):
        local = idx[(idx >= starts[gi]) & (idx < starts[gi + 1])] - starts[gi]
        rows += f.read_row_group(int(gi), columns=cols).take(local).to_pylist()
    return rows


def pct(ms) -> dict:
    a = np.asarray(ms)
    return {"p50_ms": float(np.percentile(a, 50)), "p99_ms": float(np.percentile(a, 99)), "mean_ms": float(a.mean()),
            "requests": int(a.size)}


def timed(server, rows_traffic: list[dict], requests: list[list[dict]]) -> dict:
    env_start = environment()
    for r in rows_traffic[:WARMUP]:
        server.predict([r])
    ms = []
    for r in rows_traffic[WARMUP:]:
        t = time.perf_counter()
        server.predict([r])
        ms.append(1000 * (time.perf_counter() - t))
    for req in requests[:REQ_WARMUP]:
        server.predict(req)
    ms_req, ms_model = [], []
    for req in requests[REQ_WARMUP:]:
        t = time.perf_counter()
        server.predict(req)
        ms_req.append(1000 * (time.perf_counter() - t))
    user, ad, mm, seq = server.tensors(requests[-1])
    for _ in range(REQ_N):  # the model alone, on the tensors of one request
        t = time.perf_counter()
        with server.torch.no_grad():
            server.model(user[:, 1:], ad, mm, seq)
        ms_model.append(1000 * (time.perf_counter() - t))
    env_end = environment()
    return {"one_click": pct(ms), "request_100": pct(ms_req), "request_100_model_only": pct(ms_model),
            "environment_start": env_start, "environment_end": env_end, "accepted": accepted(env_start, env_end)}


def main() -> dict:
    D, _ = R.official()
    choice = json.loads((Path("runs/e_study/config_choice.json")).read_text())
    codes = Q.token_codes()
    g1 = set(json.loads(Path("runs/e_study/group_lists.json").read_text())["G1"])
    v2 = Q.source(Q.V2)
    tokens = pq.read_table(v2, columns=["ad_feat1"]).column("ad_feat1").to_pylist()
    n = len(tokens)
    rng = np.random.default_rng(SEED)
    sparse = np.flatnonzero(np.isin(np.array(tokens, dtype=object), list(g1)))
    check_idx = np.unique(np.r_[rng.choice(n, N_UNIFORM, replace=False), rng.choice(sparse, N_SPARSE, replace=False)])
    traffic = rng.choice(n, WARMUP + REQUESTS, replace=False)
    cand = rng.choice(n, (REQ_WARMUP + REQ_N) * CANDIDATES, replace=False)
    owners = rng.choice(n, REQ_WARMUP + REQ_N, replace=False)
    cols = ServerE.columns()
    t0 = time.time()
    rows_check = dict(zip(check_idx.tolist(), read_rows(v2, check_idx, cols)))
    srt = np.sort(traffic)
    by = dict(zip(srt.tolist(), read_rows(v2, srt, cols)))
    rows_traffic = [by[int(i)] for i in traffic]
    srt = np.sort(cand)
    by = dict(zip(srt.tolist(), read_rows(v2, srt, cols)))
    rows_cand = [by[int(i)] for i in cand]
    srt = np.sort(owners)
    by = dict(zip(srt.tolist(), read_rows(v2, srt, cols)))
    user_part = D.USER_COLS + D.SEQ_COLS
    requests = []
    for k, o in enumerate(owners):
        u = {c: by[int(o)][c] for c in user_part}
        requests.append([r | u for r in rows_cand[k * CANDIDATES:(k + 1) * CANDIDATES]])
    print(f"rows read in {time.time() - t0:.0f}s", flush=True)
    vocabs = D.get_vocabs("data/raw/mac/vocabs")
    _, y2 = Q.file_columns(Q.V2, codes)
    servers = {}
    out = {"cpu": cpu_name(), "threads": 1, "check_rows": int(check_idx.size), "sparse_rows_in_check": int(np.isin(check_idx, sparse).sum()),
           "protocol": {"warmup": WARMUP, "requests": REQUESTS, "request_warmup": REQ_WARMUP, "request_n": REQ_N,
                        "candidates": CANDIDATES, "seed": SEED}, "models": {}}
    for model in ("BASE", "MoAE"):
        c = choice[model]
        rid = Q.run_id(model, c["lr"], c["lamda"], 2026)
        saved = Q.predictions(rid, Q.V2, y2)["p"]
        server = ServerE(rid, Q.RUNS / rid / "model_after_train-17.pt", vocabs, codes, threads=1)
        raw = np.array([server.raw([rows_check[i]])[0] for i in check_idx])
        served = np.array([server.predict([rows_check[i]])[0] for i in check_idx])
        _, ref = Q.load_calibrated(rid, "iii", "main", saved[check_idx], np.array([codes.get(tokens[i], -1) for i in check_idx]))
        d_raw, d_cal = float(np.abs(raw - saved[check_idx]).max()), float(np.abs(served - ref).max())
        if not (np.isfinite([d_raw, d_cal]).all() and d_raw <= NN_SINGLE_VS_BATCH_TOL and d_cal <= NN_SINGLE_VS_BATCH_TOL):
            raise CheckFailed(f"{model}: served vs saved raw {d_raw}, calibrated {d_cal} (tolerance {NN_SINGLE_VS_BATCH_TOL})")
        servers[model] = server
        out["models"][model] = {"server": server.info, "consistency": {"max_abs_diff_raw": d_raw, "max_abs_diff_calibrated": d_cal,
                                                                      "tolerance": NN_SINGLE_VS_BATCH_TOL}, "rounds": []}
        print(model, json.dumps(out["models"][model]["consistency"]), flush=True)
    for k in range(ROUNDS):  # BASE, MoAE in one round and MoAE, BASE in the next, on different requests each round
        order = ("BASE", "MoAE") if k % 2 == 0 else ("MoAE", "BASE")
        clicks = rows_traffic[WARMUP + (k * ROUND_CLICKS) % REQUESTS:][:ROUND_CLICKS]
        reqs = requests[REQ_WARMUP + (k * ROUND_REQUESTS) % REQ_N:][:ROUND_REQUESTS]
        for model in order:
            r = timed(servers[model], rows_traffic[:WARMUP] + clicks, requests[:REQ_WARMUP] + reqs)
            out["models"][model]["rounds"].append({"round": k} | r)
        print(f"round {k}", {m: (round(out["models"][m]["rounds"][-1]["one_click"]["p99_ms"], 3),
                                 round(out["models"][m]["rounds"][-1]["request_100"]["p99_ms"], 2),
                                 out["models"][m]["rounds"][-1]["accepted"]) for m in servers}, flush=True)
    for model in servers:
        rs = out["models"][model]["rounds"]
        out["models"][model]["summary"] = {
            f"{part}_{stat}": {"median": float(np.median([r[part][stat] for r in rs])),
                               "min": float(min(r[part][stat] for r in rs)), "max": float(max(r[part][stat] for r in rs))}
            for part in ("one_click", "request_100", "request_100_model_only") for stat in ("p50_ms", "p99_ms")}
        out["models"][model]["summary"]["accepted_rounds"] = sum(r["accepted"] for r in rs)
    OUT.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
