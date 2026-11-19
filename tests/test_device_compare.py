"""The device check fails on NaN, on different rows and on a difference above the tolerance."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import device_compare as C  # noqa: E402


def pair():
    rng = np.random.default_rng(0)
    base = {"user": np.arange(100), "y_last": rng.integers(0, 2, 100), "row": np.arange(100)}
    p = {f"p_{v}": rng.random(100).astype(np.float32) for v in ("last", "first", "mta", "linear")}
    return base | p, base | {k: (v + 1e-7).astype(np.float32) for k, v in p.items()}


def test_close_predictions_pass():
    g, c = pair()
    assert C.compare(g, c)["passed"]


def test_nan_in_any_column_fails():
    g, c = pair()
    c["p_mta"] = c["p_mta"].copy()
    c["p_mta"][3] = np.nan  # max() skips NaN, so this must be caught explicitly
    assert not C.compare(g, c)["passed"]


def test_other_rows_or_a_large_difference_fail():
    g, c = pair()
    assert not C.compare(g, c | {"user": c["user"][::-1]})["passed"]
    assert not C.compare(g, c | {"p_last": c["p_last"] + 1e-3})["passed"]
    assert not C.compare(g, {k: v for k, v in c.items() if k != "p_first"})["passed"]
