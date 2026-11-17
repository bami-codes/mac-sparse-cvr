"""Hard checks must stop on injected errors, not just record them."""

import numpy as np
import pytest

from cvr.checks import NUM_TOL, CheckFailed, require_close, serving_consistency


def _ok():
    rng = np.random.default_rng(0)
    codes = {"user_feat0": rng.integers(0, 9, 50), "ad_feat1": rng.integers(0, 99, 50)}
    num = rng.normal(size=(50, 8))
    p = rng.uniform(0.001, 0.2, 50)
    return codes, {k: v.copy() for k, v in codes.items()}, num, num.copy(), p, p.copy()


def test_identical_paths_pass():
    out = serving_consistency(*_ok())
    assert out["codes_mismatch"] == 0 and out["prediction_max_abs_diff"] == 0.0


def test_code_mismatch_stops():
    c_on, c_off, n_on, n_off, p_on, p_off = _ok()
    c_on["ad_feat1"][3] += 1
    with pytest.raises(CheckFailed, match="code mismatch"):
        serving_consistency(c_on, c_off, n_on, n_off, p_on, p_off)


def test_numeric_feature_error_stops():
    c_on, c_off, n_on, n_off, p_on, p_off = _ok()
    n_on[7, 2] += 10 * NUM_TOL
    with pytest.raises(CheckFailed, match="numeric features"):
        serving_consistency(c_on, c_off, n_on, n_off, p_on, p_off)


def test_prediction_error_like_the_review_injection_stops():
    c_on, c_off, n_on, n_off, p_on, p_off = _ok()
    p_on[0] += 0.0271  # size of the error a reviewer injected into a serving feature
    with pytest.raises(CheckFailed, match="calibrated predictions"):
        serving_consistency(c_on, c_off, n_on, n_off, p_on, p_off)


def test_non_finite_prediction_stops():
    c_on, c_off, n_on, n_off, p_on, p_off = _ok()
    p_on[1] = np.nan
    with pytest.raises(CheckFailed, match="finite"):
        serving_consistency(c_on, c_off, n_on, n_off, p_on, p_off)


def test_require_close_rejects_nan_and_large():
    assert require_close("x", 0.0, 1e-9) == 0.0
    with pytest.raises(CheckFailed):
        require_close("x", float("nan"), 1e-9)
    with pytest.raises(CheckFailed):
        require_close("x", 1e-6, 1e-9)
