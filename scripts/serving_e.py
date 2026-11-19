"""Study E serving: consistency and latency of BASE and MoAE (train-17, seed 2026), docs/DESIGN.md §18.12.1 (v3).

    PYTHONPATH=. uv run python scripts/serving_e.py prepare        # once, on the Mac: the rows, plan and saved values
    PYTHONPATH=. uv run python scripts/serving_e.py check NAME     # consistency on this machine
    PYTHONPATH=. uv run python scripts/serving_e.py session NAME   # one latency session, NAME = <machine>_<n>
    PYTHONPATH=. uv run python scripts/serving_e.py summary        # the sessions of every machine

prepare reads V2 once and writes runs/e_study/serving_v3/{rows.parquet, plan.npz, meta.json}: the raw rows of the
consistency check (2,000 uniform and 1,000 sparse-group rows), of the click traffic, of the candidate pool and of the
request owners, with the saved evaluation predictions of the check rows. Every later step reads only these files, the
two checkpoints and their run records, the vocabularies and the frozen calibrators, so the same rows are measured on
any machine.

check: each check row served alone must give the saved raw prediction and the frozen calibrator applied to it within
1e-6, and the shared request encoding must equal one row per candidate on 320 requests.

session: latency protocol v3 (cvr/latency_v3.py). One thread at latency priority; 10,000 clicks and 6,000 requests
of 100 candidates per model after warm-up; models in ABBA blocks, the two request paths alternating inside each block;
a load gate before each block on macOS; per request the wall time by stage, the thread CPU time, involuntary context
switches and GC runs. A synthetic request takes 100 candidates from a pool of 20,000 V2 rows (their ad and context
features and precomputed similarity sequences) and the features and purchase history of another V2 row; computing the
similarities, network, parsing and feature-store lookups are not timed.
"""

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cvr import latency_v3 as L
from cvr import pymal_run as R
from cvr import study_e_eval as Q
from cvr.checks import NN_SINGLE_VS_BATCH_TOL, CheckFailed
from cvr.serving_e import ServerE
from cvr.sysinfo import cpu_name, peak_rss_gb

SEED = 20261120
N_UNIFORM, N_SPARSE = 2000, 1000
POOL, CANDIDATES = 20000, 100
CLICK_WARMUP, CLICKS, CLICK_BLOCK = 200, 10000, 500
REQ_WARMUP, REQUESTS, REQ_BLOCK = 200, 6000, 50
SHARED_CHECK = 320
SAVED_ON = "mac"  # the saved evaluation predictions were made on this Mac's CPU, in batches of 4,096
# amendment after the Linux check failed (§18.12.1): on another machine, one row at a time is compared with the
# cross-device tolerance of study E's device check (scripts/device_compare.py); the batch path, added on every machine,
# keeps the 1e-6 tolerance. On the Linux server one MoAE row differed by 1.80e-6 from the saved value and by 1.86e-6
# from the same server's own batch result (scripts/serving_device_diff.py): x86 rounds batch size 1 differently.
CROSS_MACHINE_ROW_TOL = 1e-5
SESSIONS_PER_MACHINE = 3
DIR = Path("runs/e_study/serving_v3")
CODE = ("cvr/latency_v3.py", "cvr/serving_e.py", "scripts/serving_e.py", "scripts/serving_sessions.sh")
FIELDS = ("phase", "who", "path", "item", "block", "wall", "encode", "model", "calibrate", "cpu", "nivcsw", "gc")


def read_rows(path: Path, idx: np.ndarray, cols: list[str]) -> pa.Table:
    """Rows `idx` (sorted, unique) of a parquet file, in that order, with their row numbers as v2_row."""
    f = pq.ParquetFile(path)
    starts = np.r_[0, np.cumsum([f.metadata.row_group(i).num_rows for i in range(f.num_row_groups)])]
    parts = []
    for gi in np.unique(np.searchsorted(starts, idx, side="right") - 1):
        local = idx[(idx >= starts[gi]) & (idx < starts[gi + 1])] - starts[gi]
        parts.append(f.read_row_group(int(gi), columns=cols).take(local))
    return pa.concat_tables(parts).append_column("v2_row", pa.array(idx, pa.int64()))


def runs() -> dict[str, str]:
    choice = json.loads(Path("runs/e_study/config_choice.json").read_text())
    return {m: Q.run_id(m, choice[m]["lr"], choice[m]["lamda"], 2026) for m in L.MODELS}


def prepare() -> dict:
    codes = Q.token_codes()
    g1 = set(json.loads(Path("runs/e_study/group_lists.json").read_text())["G1"])
    v2 = Q.source(Q.V2)
    tokens = np.array(pq.read_table(v2, columns=["ad_feat1"]).column("ad_feat1").to_pylist(), dtype=object)
    n = tokens.size
    rng = np.random.default_rng(SEED)
    sparse = np.flatnonzero(np.isin(tokens, list(g1)))
    check = np.unique(np.r_[rng.choice(n, N_UNIFORM, replace=False), rng.choice(sparse, N_SPARSE, replace=False)])
    clicks = rng.choice(n, CLICK_WARMUP + CLICKS, replace=False)
    pool = rng.choice(n, POOL, replace=False)
    owners = rng.choice(n, REQ_WARMUP + REQUESTS, replace=False)
    cand = np.stack([rng.choice(POOL, CANDIDATES, replace=False) for _ in owners])  # positions in the pool
    need = np.unique(np.r_[check, clicks, pool, owners])
    t0 = time.time()
    DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(read_rows(v2, need, ServerE.columns()), DIR / "rows.parquet")
    _, y2 = Q.file_columns(Q.V2, codes)
    saved = {f"p_{m}": Q.predictions(rid, Q.V2, y2)["p"][check] for m, rid in runs().items()}
    np.savez(DIR / "plan.npz", check=check, clicks=clicks, pool=pool, owners=owners, cand=cand,
             check_group=np.array([codes.get(tokens[i], -1) for i in check]), **saved)
    meta = {"seed": SEED, "source": str(v2), "source_rows": int(n), "rows": int(need.size), "runs": runs(),
            "rows_sha256": R.sha256(DIR / "rows.parquet"), "plan_sha256": R.sha256(DIR / "plan.npz"),
            "check_rows": int(check.size), "sparse_check_rows": int(np.isin(check, sparse).sum()),
            "read_seconds": round(time.time() - t0, 1)}
    (DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_inputs() -> tuple[dict, dict, dict]:
    meta = json.loads((DIR / "meta.json").read_text())
    for f, key in (("rows.parquet", "rows_sha256"), ("plan.npz", "plan_sha256")):
        if R.sha256(DIR / f) != meta[key]:
            raise CheckFailed(f"{f} differs from the one prepared")
    t = pq.read_table(DIR / "rows.parquet")
    rows = dict(zip(t.column("v2_row").to_pylist(), t.drop(["v2_row"]).to_pylist()))
    return meta, dict(np.load(DIR / "plan.npz")), rows


def servers(meta: dict) -> dict:
    D, _ = R.official()
    vocabs = D.get_vocabs("data/raw/mac/vocabs")
    codes = Q.token_codes()
    return {m: ServerE(rid, Q.RUNS / rid / "model_after_train-17.pt", vocabs, codes, threads=1)
            for m, rid in meta["runs"].items()}


def requests_of(plan: dict, rows: dict) -> list[tuple[dict, list[dict]]]:
    D, _ = R.official()
    user_cols = D.USER_COLS + D.SEQ_COLS
    pool = [rows[int(i)] for i in plan["pool"]]
    return [({c: rows[int(o)][c] for c in user_cols}, [pool[j] for j in plan["cand"][k]])
            for k, o in enumerate(plan["owners"])]


def code_version() -> dict:
    """SHA-256 of the measuring and serving code, and the git commit when there is a repository (the Linux server
    gets a copy without .git: its files are compared with the commit's by these hashes)."""
    git = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *CODE], check=False, capture_output=True, text=True)
    return {"files_sha256": {f: R.sha256(Path(f)) for f in CODE},
            "git_commit": git.stdout.strip() if git.returncode == 0 else None,
            "code_uncommitted": bool(dirty.stdout.strip()) if git.returncode == 0 else None}


def machine() -> dict:
    import torch
    return {"cpu": cpu_name(), "platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads()}


def check(name: str) -> dict:
    meta, plan, rows = load_inputs()
    srv = servers(meta)
    reqs = requests_of(plan, rows)[:SHARED_CHECK]
    served_rows = [rows[int(i)] for i in plan["check"]]
    out = {"name": name, "machine": machine(), "code": code_version(), "models": {}}
    row_tol = NN_SINGLE_VS_BATCH_TOL if name == SAVED_ON else CROSS_MACHINE_ROW_TOL
    tols = {"max_abs_diff_raw": row_tol, "max_abs_diff_calibrated": row_tol,
            "max_abs_diff_batch_raw": NN_SINGLE_VS_BATCH_TOL, "max_abs_diff_shared_vs_per_row": NN_SINGLE_VS_BATCH_TOL}
    for m, server in srv.items():
        saved = plan[f"p_{m}"]
        raw = np.array([server.raw([r])[0] for r in served_rows])
        cal = np.array([server.predict([r])[0] for r in served_rows])
        batch = server.raw(served_rows)
        _, ref = Q.load_calibrated(meta["runs"][m], "iii", "main", saved, plan["check_group"])
        d_req = max(float(np.abs(server.predict([c | o for c in cs]) - server.predict_request(o, cs)).max())
                    for o, cs in reqs)
        res = {"max_abs_diff_raw": float(np.abs(raw - saved).max()), "max_abs_diff_calibrated": float(np.abs(cal - ref).max()),
               "max_abs_diff_batch_raw": float(np.abs(batch - saved).max()),
               "rows_raw_above_1e-6": int((np.abs(raw - saved) > NN_SINGLE_VS_BATCH_TOL).sum()),
               "max_abs_diff_shared_vs_per_row": d_req, "rows": int(raw.size), "requests": len(reqs),
               "tolerances": tols, "oov": dict(server.oov)}
        out["models"][m] = res
        bad = [k for k, t in tols.items() if not (np.isfinite(res[k]) and res[k] <= t)]
        if bad:
            raise CheckFailed(f"{m} on {name}: {bad} {res}")
    (DIR / f"check_{name}.json").write_text(json.dumps(out, indent=2))
    return out


def session(name: str) -> dict:
    import torch
    dest = DIR / "sessions" / f"{name}.json"
    if dest.exists() or dest.with_suffix(".npz").exists():
        raise CheckFailed(f"{dest} exists: a session is never measured again under the same name")
    torch.set_num_interop_threads(1)
    prio = L.set_latency_priority()  # before the servers: threads created later inherit the Linux affinity
    L.install_gc_counter()
    meta, plan, rows = load_inputs()
    srv = servers(meta)
    reqs = requests_of(plan, rows)
    clicks = [rows[int(i)] for i in plan["clicks"]]
    per_row = [[c | o for c in cs] for o, cs in reqs]  # the merge of the user's fields stays outside the timing

    def encoder(s, path: str, k: int):
        return (lambda: s.tensors(per_row[k])) if path == "per_row" else (lambda: s.request_tensors(*reqs[k]))

    for s in srv.values():  # warm-up on requests and clicks that are not timed
        for k in range(REQ_WARMUP):
            for p in L.PATHS:
                L.probe(s, encoder(s, p, k), reqs[k][1])
        for r in clicks[:CLICK_WARMUP]:
            L.probe(s, lambda r=r, s=s: s.tensors([r]), [r])
    rec = {k: [] for k in FIELDS}
    blocks = []
    env_start, t_start = L.load_sample(1.0), time.time()

    def block(phase: str, model: str) -> tuple:
        g = L.gate()
        blocks.append({"id": len(blocks), "phase": phase, "model": model, "t": round(time.time() - t_start, 2)} | g)
        return len(blocks) - 1, srv[model], L.MODELS.index(model)

    def add(*values) -> None:
        for k, v in zip(FIELDS, values):
            rec[k].append(v)

    for b in L.click_schedule(CLICKS, CLICK_BLOCK):
        bi, s, m = block("click", b["model"])
        for i in b["clicks"]:
            r = clicks[CLICK_WARMUP + i]
            add(0, m, -1, i, bi, *L.probe(s, lambda r=r, s=s: s.tensors([r]), [r]))
    for b in L.request_schedule(REQUESTS, REQ_BLOCK):
        bi, s, m = block("request", b["model"])
        for i, p in zip(b["requests"], b["paths"]):
            k = REQ_WARMUP + i
            add(1, m, L.PATHS.index(p), i, bi, *L.probe(s, encoder(s, p, k), reqs[k][1]))
    env_end = L.load_sample(1.0)
    (DIR / "sessions").mkdir(parents=True, exist_ok=True)
    np.savez(DIR / "sessions" / f"{name}.npz", **{k: np.asarray(v, dtype=np.int64) for k, v in rec.items()})
    passed = float(np.mean([b["passed"] for b in blocks]))
    out = {"name": name, "machine": machine() | prio, "code": code_version(), "peak_rss_gb": peak_rss_gb(),
           "seconds": round(time.time() - t_start, 1), "env_start": env_start, "env_end": env_end, "blocks": blocks,
           "gate_pass_share": passed, "valid": bool(passed >= L.VALID_SESSION_PASS_SHARE),
           "oov": {m: dict(s.oov) for m, s in srv.items()},
           "protocol": {"seed": SEED, "clicks": CLICKS, "click_block": CLICK_BLOCK, "requests": REQUESTS,
                        "request_block": REQ_BLOCK, "candidates": CANDIDATES, "pool": POOL,
                        "warmup_clicks": CLICK_WARMUP, "warmup_requests": REQ_WARMUP,
                        "gate_other_cpu": L.GATE_OTHER_CPU, "gate_max_wait_s": L.GATE_MAX_WAIT_S}}
    dest.write_text(json.dumps(out, indent=1))
    return {k: out[k] for k in ("name", "seconds", "gate_pass_share", "valid", "peak_rss_gb")}


def analyse(name: str) -> dict:
    z = dict(np.load(DIR / "sessions" / f"{name}.npz"))
    out = {}
    for m, model in enumerate(L.MODELS):
        out[f"{model} click"] = L.describe(z, (z["phase"] == 0) & (z["who"] == m), z["block"])
        req = (z["phase"] == 1) & (z["who"] == m)
        for p, path in enumerate(L.PATHS):
            out[f"{model} request {path}"] = L.describe(z, req & (z["path"] == p), z["block"])
        out[f"{model} shared/per_row p50"] = L.path_ratio(z, req, z["block"], L.p50)
        out[f"{model} shared/per_row p99"] = L.path_ratio(z, req, z["block"], L.p99)
    return out


def summary() -> dict:
    by_machine: dict[str, list] = {}
    for f in sorted((DIR / "sessions").glob("*.json")):
        s = json.loads(f.read_text())
        by_machine.setdefault(s["name"].rsplit("_", 1)[0], []).append(s)
    out = {}
    for mach, ss in by_machine.items():
        valid = [s for s in ss if s["valid"]][:SESSIONS_PER_MACHINE]  # the first three valid sessions
        an = {s["name"]: analyse(s["name"]) for s in valid}
        res = {"sessions": [s["name"] for s in ss], "used": [s["name"] for s in valid],
               "complete": len(valid) == SESSIONS_PER_MACHINE,
               "code_sha256": sorted({json.dumps(s.get("code", {}).get("files_sha256"), sort_keys=True) for s in valid}),
               "invalid": [s["name"] for s in ss if not s["valid"]], "machine": (valid or ss)[0]["machine"],
               "gate_pass_share": {s["name"]: s["gate_pass_share"] for s in ss}, "per_session": an}
        if len(valid) == SESSIONS_PER_MACHINE:
            first = an[valid[0]["name"]]
            res["stability"] = {f"{k} {q}": L.stability([an[s["name"]][k]["wall"][q] for s in valid])
                                for k in first if "/" not in k for q in ("p50", "p99")}
            res["stability"] |= {k: L.stability([an[s["name"]][k]["ratio"] for s in valid]) for k in first if "/" in k}
            res["effect"] = {k: L.effect([an[s["name"]][k]["ci95"] for s in valid]) for k in first if "/" in k}
        out[mach] = res
    (DIR / "summary.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1:]
    if cmd == ["prepare"]:
        print(json.dumps(prepare(), indent=1))
    elif len(cmd) == 2 and cmd[0] == "check":
        print(json.dumps(check(cmd[1]), indent=1))
    elif len(cmd) == 2 and cmd[0] == "session":
        print(json.dumps(session(cmd[1]), indent=1))
    elif cmd == ["summary"]:
        print(json.dumps({m: v.get("stability", v["sessions"]) for m, v in summary().items()}, indent=1))
    else:
        raise SystemExit(__doc__)
