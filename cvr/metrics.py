"""Evaluation metrics (docs/DESIGN.md §6).

Primary metric: macro (equal-weight) mean over G1 groups of the per-group mean LogLoss.
"""

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.metrics import roc_auc_score

EPS = 1e-7


def bce(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ece_equal_freq(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    order = np.argsort(p, kind="stable")
    parts = np.array_split(order, bins)
    return float(sum(abs(y[i].mean() - p[i].mean()) * i.size for i in parts) / y.size)


def overall(y: np.ndarray, p: np.ndarray) -> dict:
    return {"rows": int(y.size), "logloss": float(bce(y, p).mean()),
            "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < y.size else None,
            "O_over_E": float(y.sum() / p.sum()), "ece10": ece_equal_freq(y, p)}


def per_group(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> pl.DataFrame:
    df = pl.DataFrame({"g": g, "y": y, "p": p, "l": bce(y, p)})
    return df.group_by("g").agg(pl.len().alias("n"), pl.col("y").sum().alias("O"),
                                pl.col("p").sum().alias("E"), pl.col("l").mean().alias("logloss"))


def within_group_auc_macro(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> tuple[float | None, int, int]:
    """Mean of per-group AUC over groups with both classes; returns (mean, used, excluded)."""
    df = pl.DataFrame({"g": g, "y": y, "p": p})
    aucs, excluded = [], 0
    for _, part in df.group_by("g"):
        yy = part["y"].to_numpy()
        if 0 < yy.sum() < yy.size:
            aucs.append(roc_auc_score(yy, part["p"].to_numpy()))
        else:
            excluded += 1
    return (float(np.mean(aucs)) if aucs else None), len(aucs), excluded


def group_report(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> dict:
    """Primary metric and group calibration for one set of groups (e.g. all G1 rows)."""
    pg = per_group(y, p, g)
    with_pos = pg.filter(pl.col("O") > 0)
    zero = pg.filter(pl.col("O") == 0)
    auc_macro, used, excl = within_group_auc_macro(y, p, g)
    return {
        "groups": pg.height, "rows": int(y.size),
        "macro_logloss": float(pg["logloss"].mean()),
        "click_weighted_logloss": float(bce(y, p).mean()),
        "pooled_O_over_E": float(y.sum() / p.sum()),
        "median_abs_ratio_dev": float((with_pos["O"] / with_pos["E"] - 1).abs().median()) if with_pos.height else None,
        "zero_positive_groups": zero.height,
        "zero_positive_mean_E_minus_O": float((zero["E"] - zero["O"]).mean()) if zero.height else None,
        "within_group_auc_macro": auc_macro, "auc_groups_used": used, "auc_groups_excluded": excl,
    }


def paired_bootstrap_macro_diff(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, g: np.ndarray,
                                users: np.ndarray, reps: int = 1000, seed: int = 0) -> dict:
    """Macro LogLoss(A) - macro LogLoss(B) with a 95% interval from resampling USERS.

    Both models are scored on the SAME resampled users in every replicate (paired).
    A group enters a replicate's macro mean only if it has at least one resampled click.
    """
    _, ui = np.unique(users, return_inverse=True)
    _, gi = np.unique(g, return_inverse=True)
    U, G = ui.max() + 1, gi.max() + 1
    n = sparse.csr_matrix((np.ones_like(y, dtype=float), (ui, gi)), shape=(U, G))
    la = sparse.csr_matrix((bce(y, p_a), (ui, gi)), shape=(U, G))
    lb = sparse.csr_matrix((bce(y, p_b), (ui, gi)), shape=(U, G))

    def macro(w: np.ndarray, L: sparse.csr_matrix) -> np.ndarray:
        cnt, tot = n.T @ w, L.T @ w
        keep = cnt > 0
        return (tot[keep] / cnt[keep]).mean()

    point = macro(np.ones(U), la) - macro(np.ones(U), lb)
    rng = np.random.default_rng(seed)
    diffs = np.empty(reps)
    for r in range(reps):
        w = np.bincount(rng.integers(U, size=U), minlength=U).astype(float)
        diffs[r] = macro(w, la) - macro(w, lb)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"diff": float(point), "ci95": [float(lo), float(hi)], "reps": reps, "users": int(U), "groups": int(G)}
