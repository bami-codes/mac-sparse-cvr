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
    return {
        "rows": int(y.size),
        "logloss": float(bce(y, p).mean()),
        "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < y.size else None,
        "O_over_E": float(y.sum() / p.sum()),
        "ece10": ece_equal_freq(y, p),
    }


def per_group(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> pl.DataFrame:
    df = pl.DataFrame({"g": g, "y": y, "p": p, "l": bce(y, p)})
    return df.group_by("g").agg(
        pl.len().alias("n"),
        pl.col("y").sum().alias("O"),
        pl.col("p").sum().alias("E"),
        pl.col("l").mean().alias("logloss"),
    )


def grouped_auc(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> pl.DataFrame:
    """Per-group AUC from within-group ranks (Mann-Whitney), ties get average ranks.

    AUC_g = (sum of positive ranks - n_pos (n_pos + 1) / 2) / (n_pos * n_neg); null if one class only.
    """
    df = pl.DataFrame({"g": g, "y": y.astype(np.int8), "p": p})
    df = df.with_columns(pl.col("p").rank("average").over("g").alias("r"))
    agg = df.group_by("g").agg(
        pl.len().alias("n"),
        pl.col("y").sum().cast(pl.Int64).alias("n_pos"),
        (pl.col("r") * pl.col("y")).sum().alias("rank_pos"),
    )
    n_neg = pl.col("n") - pl.col("n_pos")
    return agg.with_columns(
        pl.when((pl.col("n_pos") > 0) & (n_neg > 0))
        .then((pl.col("rank_pos") - pl.col("n_pos") * (pl.col("n_pos") + 1) / 2) / (pl.col("n_pos") * n_neg))
        .otherwise(None)
        .alias("auc")
    )


def within_group_auc_macro(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> tuple[float | None, int, int]:
    """Mean of per-group AUC over groups with both classes; returns (mean, used, excluded)."""
    a = grouped_auc(y, p, g)
    used = a.filter(pl.col("auc").is_not_null())
    return (float(used["auc"].mean()) if used.height else None), used.height, a.height - used.height


def gauc(y: np.ndarray, p: np.ndarray, users: np.ndarray) -> dict:
    """MAC-style GAUC: per-user AUC weighted by the user's clicks, over users with both classes."""
    a = grouped_auc(y, p, users)
    used = a.filter(pl.col("auc").is_not_null())
    return {
        "gauc": float((used["auc"] * used["n"]).sum() / used["n"].sum()) if used.height else None,
        "users_used": used.height,
        "users_excluded": a.height - used.height,
        "click_share_used": float(used["n"].sum() / y.size),
    }


def group_report(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> dict:
    """Primary metric and group calibration for one set of groups (e.g. all G1 rows)."""
    if y.size == 0:  # fail here with a clear message, not later as float(None) inside the metric
        raise ValueError("group_report: no rows; the group definition selects no group in this data")
    pg = per_group(y, p, g)
    with_pos = pg.filter(pl.col("O") > 0)
    zero = pg.filter(pl.col("O") == 0)
    auc_macro, used, excl = within_group_auc_macro(y, p, g)
    return {
        "groups": pg.height,
        "rows": int(y.size),
        "macro_logloss": float(pg["logloss"].mean()),
        "click_weighted_logloss": float(bce(y, p).mean()),
        "pooled_O_over_E": float(y.sum() / p.sum()),
        "median_abs_ratio_dev": float((with_pos["O"] / with_pos["E"] - 1).abs().median()) if with_pos.height else None,
        # group aggregate prediction error sum|E - O| / sum O over the groups (docs/DESIGN.md §13).
        # It is not a measure of bidding loss or misallocated spend; non-zero even for a perfect model.
        "abs_conv_error_share": float((pg["E"] - pg["O"]).abs().sum() / pg["O"].sum()) if pg["O"].sum() > 0 else None,
        "zero_positive_groups": zero.height,
        "zero_positive_mean_E_minus_O": float((zero["E"] - zero["O"]).mean()) if zero.height else None,
        "within_group_auc_macro": auc_macro,
        "auc_groups_used": used,
        "auc_groups_excluded": excl,
    }


def paired_bootstrap_macro_diff(
    y: np.ndarray,
    p_a: np.ndarray | list[np.ndarray],
    p_b: np.ndarray | list[np.ndarray],
    g: np.ndarray,
    users: np.ndarray,
    reps: int = 1000,
    seed: int = 0,
) -> dict:
    """Macro LogLoss(A) - macro LogLoss(B) with a 95% interval from resampling USERS.

    p_a / p_b may be lists of seed-matched predictions (A seed k is paired with B seed k).
    Every replicate draws one set of resampled users, used for both models and all seed pairs;
    the replicate's statistic is the mean over seed pairs of the paired difference.
    A group enters a replicate's macro mean only if it has at least one resampled click.
    """
    pa = p_a if isinstance(p_a, list) else [p_a]
    pb = p_b if isinstance(p_b, list) else [p_b]
    if len(pa) != len(pb):
        raise ValueError("seed-matched prediction lists must have equal length")
    _, ui = np.unique(users, return_inverse=True)
    _, gi = np.unique(g, return_inverse=True)
    U, G = ui.max() + 1, gi.max() + 1
    n = sparse.csr_matrix((np.ones_like(y, dtype=float), (ui, gi)), shape=(U, G))
    # The macro difference is linear in the per-row loss differences, so averaging the seed pairs'
    # row differences first gives exactly the mean of the per-pair macro differences.
    row_diffs = [bce(y, a) - bce(y, b) for a, b in zip(pa, pb)]

    def as_matrix(v: np.ndarray) -> sparse.csr_matrix:
        return sparse.csr_matrix((v, (ui, gi)), shape=(U, G))

    def macro(w: np.ndarray, d: sparse.csr_matrix) -> float:
        cnt, tot = n.T @ w, d.T @ w
        keep = cnt > 0
        return float((tot[keep] / cnt[keep]).mean())

    ones = np.ones(U)
    d_mean = as_matrix(np.mean(row_diffs, axis=0))
    rng = np.random.default_rng(seed)
    diffs = np.array(
        [macro(np.bincount(rng.integers(U, size=U), minlength=U).astype(float), d_mean) for _ in range(reps)]
    )
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "diff": macro(ones, d_mean),
        "ci95": [float(lo), float(hi)],
        "reps": reps,
        "users": int(U),
        "groups": int(G),
        "seed_pairs": len(pa),
        "per_pair_diff": [macro(ones, as_matrix(r)) for r in row_diffs],
    }


def poisson_deviance(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Per-group Poisson deviance of observed positive counts against the summed predicted probabilities."""
    o, e = np.asarray(observed, float), np.asarray(expected, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(o > 0, o * np.log(o / e), 0.0)
    return 2 * (term - (o - e))
