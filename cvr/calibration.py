"""Post-hoc calibration, all fitted on V1 only.

(ii)  GlobalPlatt:       logit(p') = a * logit(p) + b
(iii) GroupShrunkPlatt:  logit(p') = (a + da_g) * logit(p) + (b + db_g), with an L2 penalty
                         lam * sum(da_g^2 + db_g^2) pulling every group toward (ii)'s (a, b)
(iv)  FreqAware:         logit(p') = w1 * logit(p) + w2 * log(1 + training clicks of g) + w0
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

EPS = 1e-7
MIN_SLOPE = 1e-3


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class GlobalPlatt:
    a: float
    b: float

    @classmethod
    def fit(cls, p: np.ndarray, y: np.ndarray) -> "GlobalPlatt":
        lr = LogisticRegression(C=1e8, solver="lbfgs", max_iter=1000, tol=1e-10).fit(logit(p)[:, None], y)
        return cls(float(lr.coef_[0, 0]), float(lr.intercept_[0]))

    def predict(self, p: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        return expit(self.a * logit(p) + self.b)


@dataclass
class GroupShrunkPlatt:
    a: float
    b: float
    lam: float
    group_ids: np.ndarray  # sorted group codes that received offsets
    da: np.ndarray
    db: np.ndarray
    n_nonpositive_slopes: int  # should always be 0 given the bounds; kept as a check
    converged: bool

    @classmethod
    def fit(cls, p: np.ndarray, y: np.ndarray, groups: np.ndarray, base: GlobalPlatt, lam: float) -> "GroupShrunkPlatt":
        z = logit(p)
        gid, gi = np.unique(groups, return_inverse=True)
        G = gid.size

        def f(theta: np.ndarray) -> tuple[float, np.ndarray]:
            da, db = theta[:G], theta[G:]
            s = (base.a + da[gi]) * z + (base.b + db[gi])
            # binary cross-entropy written with logaddexp for numerical stability
            loss = np.logaddexp(0.0, s).sum() - (y * s).sum() + lam * (da @ da + db @ db)
            r = expit(s) - y
            grad = np.concatenate([np.bincount(gi, r * z, G) + 2 * lam * da,
                                   np.bincount(gi, r, G) + 2 * lam * db])
            return loss, grad

        # a + da_g > 0 is enforced with bounds so the mapping stays increasing in every group
        # (then within-group AUC cannot change); intercept offsets are unbounded.
        bounds = [(-base.a + MIN_SLOPE, None)] * G + [(None, None)] * G
        res = minimize(f, np.zeros(2 * G), jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 1000})
        da, db = res.x[:G], res.x[G:]
        return cls(base.a, base.b, lam, gid, da, db, int((base.a + da <= 0).sum()), bool(res.success))

    def predict(self, p: np.ndarray, groups: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(self.group_ids, groups)
        pos = np.clip(pos, 0, self.group_ids.size - 1)
        known = self.group_ids[pos] == groups  # groups absent from V1 get the global (ii) mapping
        da = np.where(known, self.da[pos], 0.0)
        db = np.where(known, self.db[pos], 0.0)
        return expit((self.a + da) * logit(p) + (self.b + db))


@dataclass
class FreqAware:
    w: np.ndarray  # [w1 (logit p), w2 (log1p train clicks)]
    w0: float

    @staticmethod
    def _x(p: np.ndarray, train_clicks: np.ndarray) -> np.ndarray:
        return np.column_stack([logit(p), np.log1p(train_clicks)])

    @classmethod
    def fit(cls, p: np.ndarray, y: np.ndarray, train_clicks: np.ndarray) -> "FreqAware":
        lr = LogisticRegression(C=1e8, solver="lbfgs", max_iter=1000, tol=1e-10).fit(cls._x(p, train_clicks), y)
        return cls(lr.coef_[0].copy(), float(lr.intercept_[0]))

    def predict(self, p: np.ndarray, train_clicks: np.ndarray) -> np.ndarray:
        return expit(self._x(p, train_clicks) @ self.w + self.w0)
