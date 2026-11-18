"""Hard checks: a failed check raises and stops the run instead of only being recorded.

Tolerances are fixed in advance. Reloaded models and calibrators are expected to reproduce
predictions exactly (observed 0.0), so a tiny tolerance only absorbs float noise. The serving
path recomputes float32 similarity statistics in numpy, so numeric features get a small tolerance.
"""

import numpy as np

RELOAD_TOL = 1e-9  # max |reloaded - in-memory| prediction difference
PRED_TOL = 1e-9  # max |serving - evaluation path| calibrated prediction difference
NUM_TOL = 1e-6  # max |serving - evaluation path| numeric feature difference
# Study C networks (docs/DESIGN.md §16.8): a single-row forward and a batched forward of the SAME float32 weights
# and features differ by float rounding (5.96e-8 seen in review), so this comparison gets its own tolerance.
# Categorical codes and numeric features are compared exactly / with NUM_TOL beforehand, and reloads in the same
# batch stay at RELOAD_TOL, so a real prediction error cannot hide in this tolerance.
NN_SINGLE_VS_BATCH_TOL = 1e-6


class CheckFailed(RuntimeError):
    pass


def require_close(name: str, diff: float, tol: float) -> float:
    if not np.isfinite(diff) or diff > tol:
        raise CheckFailed(f"{name}: max abs difference {diff!r} exceeds tolerance {tol!r}")
    return diff


def require_finite_probs(name: str, p: np.ndarray) -> None:
    if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
        raise CheckFailed(f"{name}: predictions must be finite probabilities in [0, 1]")


def serving_consistency(codes_online: dict[str, np.ndarray], codes_offline: dict[str, np.ndarray],
                        num_online: np.ndarray, num_offline: np.ndarray,
                        p_online: np.ndarray, p_offline: np.ndarray) -> dict:
    """Compare the serving path with the evaluation path on the same raw rows; raise on any mismatch."""
    mismatch = {c: int((codes_online[c] != codes_offline[c]).sum()) for c in codes_offline}
    if any(mismatch.values()):
        raise CheckFailed(f"categorical code mismatch between serving and evaluation paths: "
                          f"{ {c: n for c, n in mismatch.items() if n} }")
    num_diff = float(np.abs(num_online - num_offline).max()) if num_online.size else 0.0
    require_close("numeric features (serving vs evaluation)", num_diff, NUM_TOL)
    require_finite_probs("serving predictions", p_online)
    require_finite_probs("evaluation-path predictions", p_offline)
    pred_diff = require_close("calibrated predictions (serving vs evaluation)",
                              float(np.abs(p_online - p_offline).max()), PRED_TOL)
    return {"codes_mismatch": 0, "num_max_abs_diff": num_diff, "prediction_max_abs_diff": pred_diff,
            "tolerances": {"num": NUM_TOL, "pred": PRED_TOL}}
