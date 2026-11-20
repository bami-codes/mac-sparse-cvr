"""Latency protocol v3 (cvr/latency.py): the schedule balances models and paths, and the statistics are right."""

from collections import Counter
from itertools import pairwise

import numpy as np

from cvr import latency as L


def test_request_schedule_is_balanced_abba():
    sched = L.request_schedule(6000, 50)
    assert len(sched) == 240
    for model in L.MODELS:
        served = [i for b in sched if b["model"] == model for i in b["requests"]]
        assert sorted(served) == list(range(6000))
        paths = Counter(p for b in sched if b["model"] == model for p in b["paths"])
        assert paths == {"per_row": 3000, "shared": 3000}
    assert [b["model"] for b in sched[:8]] == ["BASE", "MoAE", "MoAE", "BASE"] * 2
    for b in sched:  # paths alternate inside a block, and the first path flips from block to block
        assert all(p != q for p, q in pairwise(b["paths"]))
    firsts = [b["paths"][0] for b in sched if b["model"] == "BASE"]
    assert all(p != q for p, q in pairwise(firsts))
    # the same request is never served twice in a row by one model: consecutive blocks of a model hold other requests
    for model in L.MODELS:
        mine = [b for b in sched if b["model"] == model]
        assert all(not set(a["requests"]) & set(b["requests"]) for a, b in pairwise(mine))


def test_click_schedule_is_abba_and_complete():
    sched = L.click_schedule(10000, 500)
    assert [b["model"] for b in sched[:4]] == ["BASE", "MoAE", "MoAE", "BASE"]
    for model in L.MODELS:
        assert sorted(i for b in sched if b["model"] == model for i in b["clicks"]) == list(range(10000))


def synthetic(n_blocks=120, per_block=50, seed=0, shared_factor=0.5):
    rng = np.random.default_rng(seed)
    n = n_blocks * per_block
    path = np.tile([0, 1], n // 2)
    block = np.repeat(np.arange(n_blocks), per_block)
    base = rng.lognormal(np.log(2e6), 0.3, n)  # ns
    wall = np.where(path == 1, base * shared_factor, base)
    rec = {
        "wall": wall,
        "encode": wall * 0.5,
        "model": wall * 0.45,
        "calibrate": wall * 0.05,
        "cpu": wall * 0.98,
        "nivcsw": (rng.random(n) < 0.1).astype(int),
        "gc": (rng.random(n) < 0.2).astype(int),
        "path": path,
    }
    return rec, block


def test_describe_and_ratio_recover_known_values():
    rec, block = synthetic()
    d = L.describe(rec, rec["path"] == 0, block)
    ms = rec["wall"][rec["path"] == 0] / 1e6
    assert d["n"] == 3000 and np.isclose(d["wall"]["p99"], np.percentile(ms, 99))
    lo, hi = d["wall"]["p99_ci95"]
    assert lo <= d["wall"]["p99"] <= hi and lo < hi
    assert 0.05 < d["share_context_switched"] < 0.15 and 0.15 < d["share_gc"] < 0.25
    r = L.path_ratio(rec, np.ones(block.size, bool), block, L.p50)
    assert abs(r["ratio"] - 0.5) < 0.05 and r["ci95"][0] <= r["ratio"] <= r["ci95"][1]


def test_block_bootstrap_widens_with_correlated_blocks():
    rng = np.random.default_rng(1)
    blocks = np.repeat(np.arange(100), 50)
    shift = rng.normal(0, 1, 100)[blocks]  # a block-level load shift shared by its requests
    v = shift + rng.normal(0, 0.1, blocks.size)
    wide = L.block_bootstrap(v, blocks, np.mean)
    narrow = L.block_bootstrap(v, np.arange(blocks.size), np.mean)
    assert (wide[1] - wide[0]) > 3 * (narrow[1] - narrow[0])


def test_stability_rule():
    assert L.stability([1.0, 1.1, 1.2])["reproduced"]
    s = L.stability([1.0, 1.1, 1.3])
    assert not s["reproduced"] and s["median"] == 1.1 and np.isclose(s["max_over_min"], 1.3)


def test_effect_needs_every_session():
    assert L.effect([[0.5, 0.6], [0.55, 0.7], [0.4, 0.9]]) == "shorter"
    assert L.effect([[0.5, 0.6], [0.55, 1.01], [0.4, 0.9]]) == "undetermined"
    assert L.effect([[1.1, 1.2], [1.05, 1.3], [1.01, 1.4]]) == "longer"
