"""Check and time the served BASE and MoAE models (train-17, seed 2026). Protocol: docs/DESIGN.md §18.12.1.

    PYTHONPATH=. uv run python scripts/serving_bench.py prepare        # once, locally: the rows, plan and saved values
    PYTHONPATH=. uv run python scripts/serving_bench.py check NAME     # consistency on this machine
    PYTHONPATH=. uv run python scripts/serving_bench.py session NAME   # one latency session, NAME = <machine>_<n>
    PYTHONPATH=. uv run python scripts/serving_bench.py summary        # the sessions of every machine

prepare samples V2 once and writes runs/eval/serving/: the raw rows for the consistency check (2,000 random and
1,000 sparse-group rows), the click traffic, a pool of candidates and the request owners, plus the saved evaluation
predictions of the check rows. Later steps read only these files, the two checkpoints, the vocabularies and the
calibrators, so every machine measures the same rows.

check: every check row served on its own must match its saved prediction, raw and calibrated, and the shared request
encoding must match per-candidate encoding on 320 requests.

session: per model, 10,000 clicks and 6,000 requests of 100 candidates after warm-up, one thread at latency
priority (cvr/latency.py). A request pairs one V2 row's user features and purchase history with 100 candidates
from a pool of 20,000 V2 rows. Network, parsing, feature-store lookups and computing the similarities are not timed.
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

from cvr import evaluation, latency, pymal_run
from cvr.checks import NN_SINGLE_VS_BATCH_TOL, CheckFailed
from cvr.serving import Server
from cvr.sysinfo import cpu_name, peak_rss_gb

SEED = 20261120
N_UNIFORM, N_SPARSE = 2000, 1000
POOL, CANDIDATES = 20000, 100
CLICK_WARMUP, CLICKS, CLICK_BLOCK = 200, 10000, 500
REQ_WARMUP, REQUESTS, REQ_BLOCK = 200, 6000, 50
SHARED_CHECK = 320
SAVED_ON = "mac"  # the saved evaluation predictions were made on the local CPU, in batches of 4,096
# On another CPU, single rows get the cross-device tolerance of scripts/device_compare.py: x86 rounds batch size 1
# differently, and one MoAE row in 2,999 moved by 1.8e-6 against the same server's own batch result. The batch path
# keeps 1e-6 on every machine.
CROSS_MACHINE_ROW_TOL = 1e-5
SESSIONS_PER_MACHINE = 3
DIR = Path("runs/eval/serving")
CODE = ("cvr/latency.py", "cvr/serving.py", "scripts/serving_bench.py", "scripts/serving_sessions.sh")
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
    choice = json.loads(Path("runs/eval/config_choice.json").read_text())
    return {m: evaluation.run_id(m, choice[m]["lr"], choice[m]["lamda"], 2026) for m in latency.MODELS}


def prepare() -> dict:
    codes = evaluation.token_codes()
    g1 = set(json.loads(Path("runs/eval/group_lists.json").read_text())["G1"])
    v2 = evaluation.source(evaluation.V2)
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
    pq.write_table(read_rows(v2, need, Server.columns()), DIR / "rows.parquet")
    _, y2 = evaluation.file_columns(evaluation.V2, codes)
    saved = {f"p_{m}": evaluation.predictions(rid, evaluation.V2, y2)["p"][check] for m, rid in runs().items()}
    np.savez(
        DIR / "plan.npz",
        check=check,
        clicks=clicks,
        pool=pool,
        owners=owners,
        cand=cand,
        check_group=np.array([codes.get(tokens[i], -1) for i in check]),
        **saved,
    )
    meta = {
        "seed": SEED,
        "source": str(v2),
        "source_rows": int(n),
        "rows": int(need.size),
        "runs": runs(),
        "rows_sha256": pymal_run.sha256(DIR / "rows.parquet"),
        "plan_sha256": pymal_run.sha256(DIR / "plan.npz"),
        "check_rows": int(check.size),
        "sparse_check_rows": int(np.isin(check, sparse).sum()),
        "read_seconds": round(time.time() - t0, 1),
    }
    (DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_inputs() -> tuple[dict, dict, dict]:
    meta = json.loads((DIR / "meta.json").read_text())
    for f, key in (("rows.parquet", "rows_sha256"), ("plan.npz", "plan_sha256")):
        if pymal_run.sha256(DIR / f) != meta[key]:
            raise CheckFailed(f"{f} differs from the one prepared")
    t = pq.read_table(DIR / "rows.parquet")
    rows = dict(zip(t.column("v2_row").to_pylist(), t.drop(["v2_row"]).to_pylist()))
    return meta, dict(np.load(DIR / "plan.npz")), rows


def servers(meta: dict) -> dict:
    D, _ = pymal_run.official()
    vocabs = D.get_vocabs("data/raw/mac/vocabs")
    codes = evaluation.token_codes()
    return {
        m: Server(rid, evaluation.RUNS / rid / "model_after_train-17.pt", vocabs, codes, threads=1)
        for m, rid in meta["runs"].items()
    }


def requests_of(plan: dict, rows: dict) -> list[tuple[dict, list[dict]]]:
    D, _ = pymal_run.official()
    user_cols = D.USER_COLS + D.SEQ_COLS
    pool = [rows[int(i)] for i in plan["pool"]]
    return [
        ({c: rows[int(o)][c] for c in user_cols}, [pool[j] for j in plan["cand"][k]])
        for k, o in enumerate(plan["owners"])
    ]


def code_version() -> dict:
    """SHA-256 of the measuring and serving code, and the git commit when there is a repository (the Linux server
    gets a copy without .git: its files are compared with the commit's by these hashes)."""
    git = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *CODE], check=False, capture_output=True, text=True)
    return {
        "files_sha256": {f: pymal_run.sha256(Path(f)) for f in CODE},
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "code_uncommitted": bool(dirty.stdout.strip()) if git.returncode == 0 else None,
    }


def machine() -> dict:
    import torch

    return {
        "cpu": cpu_name(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


def check(name: str) -> dict:
    meta, plan, rows = load_inputs()
    srv = servers(meta)
    reqs = requests_of(plan, rows)[:SHARED_CHECK]
    served_rows = [rows[int(i)] for i in plan["check"]]
    out = {"name": name, "machine": machine(), "code": code_version(), "models": {}}
    row_tol = NN_SINGLE_VS_BATCH_TOL if name == SAVED_ON else CROSS_MACHINE_ROW_TOL
    tols = {
        "max_abs_diff_raw": row_tol,
        "max_abs_diff_calibrated": row_tol,
        "max_abs_diff_batch_raw": NN_SINGLE_VS_BATCH_TOL,
        "max_abs_diff_shared_vs_per_row": NN_SINGLE_VS_BATCH_TOL,
    }
    for m, server in srv.items():
        saved = plan[f"p_{m}"]
        raw = np.array([server.raw([r])[0] for r in served_rows])
        cal = np.array([server.predict([r])[0] for r in served_rows])
        batch = server.raw(served_rows)
        _, ref = evaluation.load_calibrated(meta["runs"][m], "iii", "main", saved, plan["check_group"])
        d_req = max(
            float(np.abs(server.predict([c | o for c in cs]) - server.predict_request(o, cs)).max()) for o, cs in reqs
        )
        res = {
            "max_abs_diff_raw": float(np.abs(raw - saved).max()),
            "max_abs_diff_calibrated": float(np.abs(cal - ref).max()),
            "max_abs_diff_batch_raw": float(np.abs(batch - saved).max()),
            "rows_raw_above_1e-6": int((np.abs(raw - saved) > NN_SINGLE_VS_BATCH_TOL).sum()),
            "max_abs_diff_shared_vs_per_row": d_req,
            "rows": int(raw.size),
            "requests": len(reqs),
            "tolerances": tols,
            "oov": dict(server.oov),
        }
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
    prio = latency.set_latency_priority()  # before the servers: threads created later inherit the Linux affinity
    latency.install_gc_counter()
    meta, plan, rows = load_inputs()
    srv = servers(meta)
    reqs = requests_of(plan, rows)
    clicks = [rows[int(i)] for i in plan["clicks"]]
    per_row = [[c | o for c in cs] for o, cs in reqs]  # the merge of the user's fields stays outside the timing

    def encoder(s, path: str, k: int):
        return (lambda: s.tensors(per_row[k])) if path == "per_row" else (lambda: s.request_tensors(*reqs[k]))

    for s in srv.values():  # warm-up on requests and clicks that are not timed
        for k in range(REQ_WARMUP):
            for p in latency.PATHS:
                latency.probe(s, encoder(s, p, k), reqs[k][1])
        for r in clicks[:CLICK_WARMUP]:
            latency.probe(s, lambda r=r, s=s: s.tensors([r]), [r])
    rec = {k: [] for k in FIELDS}
    blocks = []
    env_start, t_start = latency.load_sample(1.0), time.time()

    def block(phase: str, model: str) -> tuple:
        g = latency.gate()
        blocks.append({"id": len(blocks), "phase": phase, "model": model, "t": round(time.time() - t_start, 2)} | g)
        return len(blocks) - 1, srv[model], latency.MODELS.index(model)

    def add(*values) -> None:
        for k, v in zip(FIELDS, values):
            rec[k].append(v)

    for b in latency.click_schedule(CLICKS, CLICK_BLOCK):
        bi, s, m = block("click", b["model"])
        for i in b["clicks"]:
            r = clicks[CLICK_WARMUP + i]
            add(0, m, -1, i, bi, *latency.probe(s, lambda r=r, s=s: s.tensors([r]), [r]))
    for b in latency.request_schedule(REQUESTS, REQ_BLOCK):
        bi, s, m = block("request", b["model"])
        for i, p in zip(b["requests"], b["paths"]):
            k = REQ_WARMUP + i
            add(1, m, latency.PATHS.index(p), i, bi, *latency.probe(s, encoder(s, p, k), reqs[k][1]))
    env_end = latency.load_sample(1.0)
    (DIR / "sessions").mkdir(parents=True, exist_ok=True)
    np.savez(DIR / "sessions" / f"{name}.npz", **{k: np.asarray(v, dtype=np.int64) for k, v in rec.items()})
    passed = float(np.mean([b["passed"] for b in blocks]))
    out = {
        "name": name,
        "machine": machine() | prio,
        "code": code_version(),
        "peak_rss_gb": peak_rss_gb(),
        "seconds": round(time.time() - t_start, 1),
        "env_start": env_start,
        "env_end": env_end,
        "blocks": blocks,
        "gate_pass_share": passed,
        "valid": bool(passed >= latency.VALID_SESSION_PASS_SHARE),
        "oov": {m: dict(s.oov) for m, s in srv.items()},
        "protocol": {
            "seed": SEED,
            "clicks": CLICKS,
            "click_block": CLICK_BLOCK,
            "requests": REQUESTS,
            "request_block": REQ_BLOCK,
            "candidates": CANDIDATES,
            "pool": POOL,
            "warmup_clicks": CLICK_WARMUP,
            "warmup_requests": REQ_WARMUP,
            "gate_other_cpu": latency.GATE_OTHER_CPU,
            "gate_max_wait_s": latency.GATE_MAX_WAIT_S,
        },
    }
    dest.write_text(json.dumps(out, indent=1))
    return {k: out[k] for k in ("name", "seconds", "gate_pass_share", "valid", "peak_rss_gb")}


def analyse(name: str) -> dict:
    z = dict(np.load(DIR / "sessions" / f"{name}.npz"))
    out = {}
    for m, model in enumerate(latency.MODELS):
        out[f"{model} click"] = latency.describe(z, (z["phase"] == 0) & (z["who"] == m), z["block"])
        req = (z["phase"] == 1) & (z["who"] == m)
        for p, path in enumerate(latency.PATHS):
            out[f"{model} request {path}"] = latency.describe(z, req & (z["path"] == p), z["block"])
        out[f"{model} shared/per_row p50"] = latency.path_ratio(z, req, z["block"], latency.p50)
        out[f"{model} shared/per_row p99"] = latency.path_ratio(z, req, z["block"], latency.p99)
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
        res = {
            "sessions": [s["name"] for s in ss],
            "used": [s["name"] for s in valid],
            "complete": len(valid) == SESSIONS_PER_MACHINE,
            "code_sha256": sorted({json.dumps(s.get("code", {}).get("files_sha256"), sort_keys=True) for s in valid}),
            "invalid": [s["name"] for s in ss if not s["valid"]],
            "machine": (valid or ss)[0]["machine"],
            "gate_pass_share": {s["name"]: s["gate_pass_share"] for s in ss},
            "per_session": an,
        }
        if len(valid) == SESSIONS_PER_MACHINE:
            first = an[valid[0]["name"]]
            res["stability"] = {
                f"{k} {q}": latency.stability([an[s["name"]][k]["wall"][q] for s in valid])
                for k in first
                if "/" not in k
                for q in ("p50", "p99")
            }
            res["stability"] |= {
                k: latency.stability([an[s["name"]][k]["ratio"] for s in valid]) for k in first if "/" in k
            }
            res["effect"] = {k: latency.effect([an[s["name"]][k]["ci95"] for s in valid]) for k in first if "/" in k}
        out[mach] = res
    (DIR / "summary.json").write_text(json.dumps(out, indent=1))
    return out


def render(machine: str) -> str:
    """The latency tables of docs/results/serving.md for one machine, from summary.json (nothing copied by hand)."""
    r = json.loads((DIR / "summary.json").read_text())[machine]
    if not r.get("complete"):
        raise CheckFailed(f"{machine}: {len(r['used'])} valid sessions, three are needed")
    st, names = r["stability"], r["used"]
    paths = {
        "click": "클릭 한 건",
        "request per_row": "요청 (후보 100개), 후보마다 인코딩",
        "request shared": "요청 (후보 100개), 사용자 공유 인코딩",
    }

    def span(v: dict) -> str:
        return f"{v['median']:.2f} ({min(v['sessions']):.2f}–{max(v['sessions']):.2f})"

    out = ["| 모델 | 경로 | p50 | p99 |", "|---|---|---|---|"]
    for model in latency.MODELS:
        for k, label in paths.items():
            out.append(f"| {model} | {label} | {span(st[f'{model} {k} p50'])} | {span(st[f'{model} {k} p99'])} |")
    worst = max(v["max_over_min"] for k, v in st.items() if "/" not in k)
    out += [
        "",
        f"세 세션의 중앙값(범위), ms. 모든 값이 세 세션 사이에서 {worst:.2f}배 안입니다.",
        "",
        "| 모델 | 공유 ÷ 후보마다, p50 | 공유 ÷ 후보마다, p99 |",
        "|---|---|---|",
    ]
    for model in latency.MODELS:
        cells = []
        for q in ("p50", "p99"):
            k = f"{model} shared/per_row {q}"
            lo = min(r["per_session"][n][k]["ci95"][0] for n in names)
            hi = max(r["per_session"][n][k]["ci95"][1] for n in names)
            cells.append(f"{st[k]['median']:.2f} (세 세션의 95% 구간 {lo:.2f}–{hi:.2f})")
        out.append(f"| {model} | {cells[0]} | {cells[1]} |")
    out += ["", "| 모델 | 경로 | 인코딩 | 모델 계산 | 보정 |", "|---|---|---|---|---|"]
    for model in latency.MODELS:
        for k, label in paths.items():
            ds = [r["per_session"][n][f"{model} {k}"] for n in names]
            cells = [
                f"{np.median([d['stages_p50'][x] for d in ds]):.2f} / {np.median([d['stages_p99'][x] for d in ds]):.2f}"
                for x in ("encode", "model", "calibrate")
            ]
            out.append(f"| {model} | {label} | {' | '.join(cells)} |")
    switched = [
        r["per_session"][n][f"{m} {k}"]["share_context_switched"] for n in names for m in latency.MODELS for k in paths
    ]
    note = (
        f"단계별 p50 / p99, ms (세 세션의 중앙값). 측정 중 문맥 전환이 일어난 요청은 경로마다 "
        f"{min(switched):.1%}–{max(switched):.1%}였습니다."
    )
    out += ["", note]
    return "\n".join(out)


if __name__ == "__main__":
    cmd = sys.argv[1:]
    if cmd == ["prepare"]:
        print(json.dumps(prepare(), indent=1))
    elif len(cmd) == 2 and cmd[0] == "check":
        print(json.dumps(check(cmd[1]), indent=1))
    elif len(cmd) == 2 and cmd[0] == "session":
        print(json.dumps(session(cmd[1]), indent=1))
    elif len(cmd) == 2 and cmd[0] == "render":
        print(render(cmd[1]))
    elif cmd == ["summary"]:
        print(json.dumps({m: v.get("stability", v["sessions"]) for m, v in summary().items()}, indent=1))
    else:
        raise SystemExit(__doc__)
