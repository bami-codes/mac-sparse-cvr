"""Latency measurement: schedule, per-request probes, load gate and analysis (protocol in docs/DESIGN.md §18.12.1).

Models alternate in blocks in ABBA order (BASE, MoAE, MoAE, BASE, ...) so a slow drift in machine load hits both
equally. Inside a block the two request paths alternate, and the first path flips every block. No model serves the
same request twice in a row, because the second run would find its embedding rows already in cache.
"""

import gc
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

PATHS = ("per_row", "shared")
MODELS = ("BASE", "MoAE")
GATE_OTHER_CPU = 450.0  # % of one core, macOS only; the idle baseline was 369% (median), 485% (p90)
GATE_SAMPLE_S, GATE_RETRY_S, GATE_MAX_WAIT_S = 0.5, 2.0, 30.0
VALID_SESSION_PASS_SHARE = 0.9  # share of blocks that must pass the gate for a session to count
STABLE_RATIO = 1.25  # a value is "reproduced" when max / min over the sessions stays within this
P_CORES_M4 = (6, 7, 8, 9)  # performance cores of the M4 used here; 0-5 are efficiency cores


def abba(n_pairs: int) -> list[str]:
    """Model order over 2 * n_pairs blocks: BASE MoAE, MoAE BASE, BASE MoAE, ..."""
    out = []
    for k in range(n_pairs):
        out += list(MODELS) if k % 2 == 0 else list(MODELS[::-1])
    return out


def request_schedule(n_requests: int, block: int) -> list[dict]:
    """Blocks of `block` consecutive requests, each block served by both models (ABBA); path of request i in block b of
    model m: shared when (i + b + m) is even. Every request is served once by each model."""
    if n_requests % block or block % 2:
        raise ValueError("the requests must fill whole blocks of an even size")
    out = []
    for b in range(n_requests // block):
        order = MODELS if b % 2 == 0 else MODELS[::-1]
        for model in order:
            m = MODELS.index(model)
            idx = list(range(b * block, (b + 1) * block))
            out.append({"model": model, "block": b, "requests": idx,
                        "paths": [PATHS[1] if (i + b + m) % 2 == 0 else PATHS[0] for i in idx]})
    return out


def click_schedule(n_clicks: int, block: int) -> list[dict]:
    if n_clicks % block:
        raise ValueError("the clicks must fill whole blocks")
    out = []
    for b in range(n_clicks // block):
        for model in (MODELS if b % 2 == 0 else MODELS[::-1]):
            out.append({"model": model, "block": b, "clicks": list(range(b * block, (b + 1) * block))})
    return out


# ------------------------------------------------------------------------------------------- the machine and the probes
def set_latency_priority() -> dict:
    """Give the measuring thread a core of its own, as far as the OS allows.

    macOS: QoS user-interactive, which keeps the thread on performance cores ahead of default-QoS work.
    Linux: pin the process to one CPU of its allowed set, the usual setup for a latency-critical server."""
    if sys.platform == "darwin":
        import ctypes
        lib = ctypes.CDLL("/usr/lib/libSystem.dylib")
        if lib.pthread_set_qos_class_self_np(0x21, 0) != 0:
            raise RuntimeError("could not set QoS user-interactive")
        return {"priority": "macOS QoS user-interactive (0x21)", "qos_class_self": int(lib.qos_class_self())}
    if sys.platform.startswith("linux"):
        allowed = sorted(os.sched_getaffinity(0))
        cpu = max(allowed)
        os.sched_setaffinity(0, {cpu})
        sib = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        return {"priority": f"Linux affinity to CPU {cpu}", "affinity": sorted(os.sched_getaffinity(0)),
                "container_cpus": allowed, "host_cpus": os.cpu_count(),
                "pinned_cpu_siblings": sib.read_text().strip() if sib.exists() else None}
    raise RuntimeError(f"no latency priority defined for {sys.platform}")


RUSAGE = resource.RUSAGE_THREAD if hasattr(resource, "RUSAGE_THREAD") else resource.RUSAGE_SELF
GC_RUNS = [0]


def _count_gc(phase: str, info: dict) -> None:
    if phase == "start":
        GC_RUNS[0] += 1


def install_gc_counter() -> None:
    if _count_gc not in gc.callbacks:
        gc.callbacks.append(_count_gc)


def probe(server, encode, cands: list[dict]) -> tuple[int, int, int, int, int, int, int]:
    """Time one request through the server's own steps.

    Returns wall time (total, encoding, model, calibration) and thread CPU time in ns, plus the involuntary context
    switches and GC runs that happened during it. The rusage calls sit outside the timed span."""
    r0, g0 = resource.getrusage(RUSAGE).ru_nivcsw, GC_RUNS[0]
    c0 = time.thread_time_ns()
    t0 = time.perf_counter_ns()
    x = encode()
    t1 = time.perf_counter_ns()
    p = server.score(x)
    t2 = time.perf_counter_ns()
    server.calibrate(p, cands)
    t3 = time.perf_counter_ns()
    c1 = time.thread_time_ns()
    return t3 - t0, t1 - t0, t2 - t1, t3 - t2, c1 - c0, resource.getrusage(RUSAGE).ru_nivcsw - r0, GC_RUNS[0] - g0


def load_sample(seconds: float) -> dict:
    """CPU use of every process but this one over `seconds`. Inside a Linux container this covers the whole host."""
    import psutil
    me = psutil.Process()
    a = me.cpu_times()
    per = psutil.cpu_percent(interval=seconds, percpu=True)
    b = me.cpu_times()
    own = 100 * ((b.user + b.system) - (a.user + a.system)) / seconds
    out = {"other_cpu_percent": float(sum(per) - own), "per_core": [float(x) for x in per]}
    if sys.platform == "darwin" and len(per) == 10:
        out["p_cores_percent"] = float(sum(per[i] for i in P_CORES_M4))
    return out


def gate() -> dict:
    """Hold a block until the machine is quiet enough, then return the load sample.

    macOS waits until other processes use at most GATE_OTHER_CPU, for up to GATE_MAX_WAIT_S, and then runs anyway
    with the block marked as failed. Linux only records the sample: a container sees the whole host, so the same
    threshold would mean something else."""
    t0 = time.monotonic()
    while True:
        s = load_sample(GATE_SAMPLE_S)
        if sys.platform != "darwin":
            return s | {"gated": False, "passed": True, "waited_s": 0.0}
        if s["other_cpu_percent"] <= GATE_OTHER_CPU:
            return s | {"gated": True, "passed": True, "waited_s": time.monotonic() - t0}
        if time.monotonic() - t0 > GATE_MAX_WAIT_S:
            return s | {"gated": True, "passed": False, "waited_s": time.monotonic() - t0}
        time.sleep(GATE_RETRY_S)


# -------------------------------------------------------------------------------------------------------- the analysis
def block_bootstrap(values: np.ndarray, blocks: np.ndarray, stat, reps: int = 1000, seed: int = 20261119) -> list[float]:
    """95% interval of stat(values) when whole blocks are resampled with replacement (requests in one block share the
    machine's state, so they are not independent)."""
    ids = np.unique(blocks)
    groups = [values[blocks == b] for b in ids]
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for r in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        out[r] = stat(np.concatenate([groups[i] for i in pick]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def p50(a): return float(np.percentile(a, 50))
def p99(a): return float(np.percentile(a, 99))
def p999(a): return float(np.percentile(a, 99.9))


def describe(rec: dict, mask: np.ndarray, blocks: np.ndarray) -> dict:
    """Percentiles (ms) of one condition with block-bootstrap intervals, the stages, the CPU time and the share of
    requests disturbed by a context switch or a GC run."""
    ms = {k: rec[k][mask] / 1e6 for k in ("wall", "encode", "model", "calibrate", "cpu")}
    b = blocks[mask]
    quiet = (rec["nivcsw"][mask] == 0) & (rec["gc"][mask] == 0)
    out = {"n": int(mask.sum()),
           "wall": {"p50": p50(ms["wall"]), "p50_ci95": block_bootstrap(ms["wall"], b, p50),
                    "p99": p99(ms["wall"]), "p99_ci95": block_bootstrap(ms["wall"], b, p99),
                    "p999": p999(ms["wall"]), "mean": float(ms["wall"].mean())},
           "cpu": {"p50": p50(ms["cpu"]), "p99": p99(ms["cpu"])},
           "stages_p50": {k: p50(ms[k]) for k in ("encode", "model", "calibrate")},
           "stages_p99": {k: p99(ms[k]) for k in ("encode", "model", "calibrate")},
           "share_context_switched": float((rec["nivcsw"][mask] > 0).mean()),
           "share_gc": float((rec["gc"][mask] > 0).mean()),
           "undisturbed": {"n": int(quiet.sum()), "p99": p99(ms["wall"][quiet]) if quiet.sum() >= 100 else None}}
    return out


def path_ratio(rec: dict, model_mask: np.ndarray, blocks: np.ndarray, stat) -> dict:
    """stat(shared) / stat(per_row) for one model, with a block-bootstrap interval (a block holds both paths)."""
    wall = rec["wall"] / 1e6
    shared = rec["path"] == PATHS.index("shared")
    ids = np.unique(blocks[model_mask])
    groups = [(wall[model_mask & (blocks == b) & shared], wall[model_mask & (blocks == b) & ~shared]) for b in ids]
    point = stat(np.concatenate([g[0] for g in groups])) / stat(np.concatenate([g[1] for g in groups]))
    rng = np.random.default_rng(20261119)
    reps = np.empty(1000)
    for r in range(reps.size):
        pick = rng.integers(0, len(groups), len(groups))
        reps[r] = stat(np.concatenate([groups[i][0] for i in pick])) / stat(np.concatenate([groups[i][1] for i in pick]))
    return {"ratio": float(point), "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]}


def stability(values: list[float]) -> dict:
    """Over the sessions of one machine: reproduced when max / min <= STABLE_RATIO; the reported value is the median."""
    v = np.asarray(values, dtype=float)
    ratio = float(v.max() / v.min())
    return {"sessions": [float(x) for x in v], "median": float(np.median(v)), "max_over_min": ratio,
            "reproduced": bool(ratio <= STABLE_RATIO)}


def effect(cis: list[list[float]]) -> str:
    """Shared / per-row ratio over the sessions of one machine: "shorter" when every session's 95% interval lies below
    1, "longer" when every one lies above 1, otherwise "undetermined"."""
    if all(hi < 1 for _, hi in cis):
        return "shorter"
    if all(lo > 1 for lo, _ in cis):
        return "longer"
    return "undetermined"
