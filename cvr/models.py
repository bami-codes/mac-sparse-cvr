"""Base models: M0 (empirical-Bayes smoothed group rate), M1 (logistic regression), M2 (LightGBM)."""

from dataclasses import dataclass

import lightgbm as lgb
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
        g = train.group_by(key).agg(pl.len().alias("n"), pl.col("y").sum().alias("k"))
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
    mean: np.ndarray
    std: np.ndarray
    model: LogisticRegression

    @staticmethod
    def _design(df: pl.DataFrame, offsets: dict[str, int], n_onehot: int,
                mean: np.ndarray, std: np.ndarray) -> sparse.csr_matrix:
        """Exact one-hot of vocabulary codes (column offsets) + standardized numeric inputs."""
        n = df.height
        cols = [df[c].to_numpy().astype(np.int64) + offsets[c] for c in LR_ONEHOT]
        num = (df.select(NUM_INPUTS).to_numpy().astype(np.float64) - mean) / std
        idx = np.column_stack(cols + [np.tile(np.arange(len(NUM_INPUTS)) + n_onehot, (n, 1))])
        val = np.column_stack([np.ones((n, len(cols))), num])
        k = idx.shape[1]
        return sparse.csr_matrix((val.ravel(), idx.ravel(), np.arange(0, n * k + 1, k)),
                                 shape=(n, n_onehot + len(NUM_INPUTS)))

    @classmethod
    def fit(cls, train: pl.DataFrame, vocab_sizes: dict[str, int], C: float, max_iter: int) -> "LogReg":
        offsets, pos = {}, 0
        for c in LR_ONEHOT:
            offsets[c], pos = pos, pos + vocab_sizes[c]
        num = train.select(NUM_INPUTS).to_numpy().astype(np.float64)
        mean, std = num.mean(0), num.std(0) + 1e-9
        X = cls._design(train, offsets, pos, mean, std)
        model = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter, tol=1e-5)
        model.fit(X, train["y"].to_numpy())
        return cls(offsets, pos, mean, std, model)

    @property
    def converged(self) -> bool:
        return int(self.model.n_iter_[0]) < self.model.max_iter

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._design(df, self.offsets, self.n_onehot, self.mean, self.std))[:, 1]


# ---------------------------------------------------------------- M2: LightGBM
@dataclass
class GBDT:
    booster: lgb.Booster
    inputs: list[str]

    @classmethod
    def fit(cls, train: pl.DataFrame, params: dict, seed: int, num_boost_round: int,
            early_stopping_rounds: int, inputs: list[str] | None = None) -> "GBDT":
        """Early stopping uses ~10% of TRAINING users (second key digit == 0), never V1/V2."""
        inputs = inputs or (CAT_INPUTS + NUM_INPUTS)
        assert_no_label_inputs(inputs)
        cat_idx = [i for i, c in enumerate(inputs) if c in CAT_INPUTS]
        es = ((train["user_key"] // 100) % 10 == 0).to_numpy()
        X, y = train.select(inputs).to_numpy().astype(np.float64), train["y"].to_numpy()
        p = dict(params, seed=seed, objective="binary", deterministic=True, force_row_wise=True, verbose=-1)
        dtr = lgb.Dataset(X[~es], y[~es], categorical_feature=cat_idx, free_raw_data=True)
        des = lgb.Dataset(X[es], y[es], categorical_feature=cat_idx, reference=dtr)
        booster = lgb.train(p, dtr, num_boost_round=num_boost_round, valid_sets=[des],
                            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
        return cls(booster, inputs)

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        X = df.select(self.inputs).to_numpy().astype(np.float64)
        return self.booster.predict(X, num_iteration=self.booster.best_iteration)
