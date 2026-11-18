"""Base models: M0 (empirical-Bayes smoothed group rate), M1 (logistic regression), M2 (LightGBM).

LightGBM is imported only inside GBDT.fit: merely loading its OpenMP runtime makes a later PyTorch
call in the same process crash on this machine, so modules that do not train LightGBM must not load it.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.special import betaln
from sklearn.linear_model import LogisticRegression

from .schema import CAT_INPUTS, LR_ONEHOT, NUM_INPUTS, assert_no_label_inputs

assert_no_label_inputs(CAT_INPUTS + NUM_INPUTS + LR_ONEHOT)


# ---------------------------------------------------------------- M0: smoothed group rate
@dataclass
class SmoothedRate:
    key: str
    p0: float  # training positive rate (prior mean)
    m: float  # prior strength in pseudo-clicks, fitted by beta-binomial marginal likelihood
    table: pl.DataFrame  # key -> smoothed rate

    @classmethod
    def fit(cls, train: pl.DataFrame, key: str) -> "SmoothedRate":
        # sorted: polars group_by order varies between runs, which changed the summation order of the likelihood
        # and moved the fitted m in its 10th digit (docs/DESIGN.md §15.2)
        g = train.group_by(key).agg(pl.len().alias("n"), pl.col("y").sum().alias("k")).sort(key)
        n, k = g["n"].to_numpy().astype(float), g["k"].to_numpy().astype(float)
        p0 = k.sum() / n.sum()

        def neg_loglik(log_m: float) -> float:
            a, b = np.exp(log_m) * p0, np.exp(log_m) * (1 - p0)
            return -(betaln(k + a, n - k + b) - betaln(a, b)).sum()

        m = float(np.exp(minimize_scalar(neg_loglik, bounds=(0.0, np.log(1e7)), method="bounded").x))
        rate = (k + m * p0) / (n + m)
        return cls(key, p0, m, pl.DataFrame({key: g[key], "rate": rate}))

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        out = df.select(self.key).join(self.table, on=self.key, how="left", maintain_order="left")
        return out["rate"].fill_null(self.p0).to_numpy()


# ---------------------------------------------------------------- M1: logistic regression
@dataclass
class LogReg:
    offsets: dict[str, int]
    n_onehot: int
    num_inputs: list[str]
    mean: np.ndarray
    std: np.ndarray
    model: LogisticRegression

    def _design(self, df: pl.DataFrame) -> sparse.csr_matrix:
        """Exact one-hot of vocabulary codes (column offsets) + standardized numeric inputs."""
        n, k_num = df.height, len(self.num_inputs)
        cols = [df[c].to_numpy().astype(np.int64) + self.offsets[c] for c in LR_ONEHOT]
        parts_idx, parts_val = cols, [np.ones((n, len(cols)))]
        if k_num:
            num = (df.select(self.num_inputs).to_numpy().astype(np.float64) - self.mean) / self.std
            parts_idx = cols + [np.tile(np.arange(k_num) + self.n_onehot, (n, 1))]
            parts_val = parts_val + [num]
        idx, val = np.column_stack(parts_idx), np.column_stack(parts_val)
        k = idx.shape[1]
        return sparse.csr_matrix((val.ravel(), idx.ravel(), np.arange(0, n * k + 1, k)),
                                 shape=(n, self.n_onehot + k_num))

    @classmethod
    def fit(cls, train: pl.DataFrame, vocab_sizes: dict[str, int], C: float, max_iter: int,
            num_inputs: list[str] = NUM_INPUTS) -> "LogReg":
        assert_no_label_inputs(LR_ONEHOT + num_inputs)
        offsets, pos = {}, 0
        for c in LR_ONEHOT:
            offsets[c], pos = pos, pos + vocab_sizes[c]
        num = train.select(num_inputs).to_numpy().astype(np.float64) if num_inputs else np.zeros((1, 0))
        m = cls(offsets, pos, list(num_inputs), num.mean(0), num.std(0) + 1e-9,
                LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter, tol=1e-5))
        m.model.fit(m._design(train), train["y"].to_numpy())
        return m

    @property
    def converged(self) -> bool:
        return int(self.model.n_iter_[0]) < self.model.max_iter

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._design(df))[:, 1]


# ---------------------------------------------------------------- M2: LightGBM
@dataclass
class GBDT:
    booster: Any  # lightgbm.Booster
    inputs: list[str]
    rounds: int  # chosen by early stopping on held-out TRAINING users, then refit on all training rows

    @classmethod
    def fit(cls, train: pl.DataFrame, params: dict, seed: int, num_boost_round: int,
            early_stopping_rounds: int, inputs: list[str] | None = None, label: str = "y") -> "GBDT":
        """Two stages so M2 sees the same training rows as M1:
        1) early stopping on ~10% of TRAINING users (second key digit == 0) picks the rounds;
        2) the model is refit on ALL training rows with that number of rounds. V1/V2 are never used.
        `label` is "y" for every CVR model; the auxiliary model of cvr/stacking.py uses "label_mta".
        """
        import lightgbm as lgb

        inputs = inputs or (CAT_INPUTS + NUM_INPUTS)
        assert_no_label_inputs(inputs)
        cat_idx = [i for i, c in enumerate(inputs) if c in CAT_INPUTS]
        es = ((train["user_key"] // 100) % 10 == 0).to_numpy()
        X, y = train.select(inputs).to_numpy().astype(np.float64), train[label].to_numpy()
        p = dict(params, seed=seed, objective="binary", deterministic=True, force_row_wise=True, verbose=-1)
        dtr = lgb.Dataset(X[~es], y[~es], categorical_feature=cat_idx, free_raw_data=True)
        des = lgb.Dataset(X[es], y[es], categorical_feature=cat_idx, reference=dtr)
        probe = lgb.train(p, dtr, num_boost_round=num_boost_round, valid_sets=[des],
                          callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
        rounds = max(1, probe.best_iteration)
        full = lgb.train(p, lgb.Dataset(X, y, categorical_feature=cat_idx, free_raw_data=True),
                         num_boost_round=rounds)
        return cls(full, inputs, rounds)

    @classmethod
    def fit_fixed(cls, train: pl.DataFrame, params: dict, seed: int, rounds: int, inputs: list[str],
                  label: str = "y") -> "GBDT":
        """Train on all rows with a GIVEN number of rounds (no early stopping), for ablation protocols where
        every variant must share the same training budget so differences come from the inputs only."""
        import lightgbm as lgb

        assert_no_label_inputs(inputs)
        cat_idx = [i for i, c in enumerate(inputs) if c in CAT_INPUTS]
        X, y = train.select(inputs).to_numpy().astype(np.float64), train[label].to_numpy()
        p = dict(params, seed=seed, objective="binary", deterministic=True, force_row_wise=True, verbose=-1)
        booster = lgb.train(p, lgb.Dataset(X, y, categorical_feature=cat_idx, free_raw_data=True), num_boost_round=rounds)
        return cls(booster, inputs, rounds)

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return self.booster.predict(df.select(self.inputs).to_numpy().astype(np.float64))
