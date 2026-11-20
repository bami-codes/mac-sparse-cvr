"""Post-hoc calibration of predicted probabilities, fitted on a held-out day.

(ii)  GlobalPlatt:       logit(p') = a * logit(p) + b
(iii) GroupShrunkPlatt:  logit(p') = (a + da_g) * logit(p) + (b + db_g), with an L2 penalty
                         lam * sum(da_g^2 + db_g^2) pulling every group toward (ii)'s (a, b)
(iv)  FreqAware:         logit(p') = w1 * logit(p) + w2 * log(1 + training clicks of g) + w0

scikit-learn is imported inside fit() only. Processes that just predict, such as the PyTorch server, then avoid
loading a second OpenMP runtime.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import expit

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
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1e8, solver="lbfgs", max_iter=1000, tol=1e-10).fit(logit(p)[:, None], y)
        return cls(float(lr.coef_[0, 0]), float(lr.intercept_[0]))

    def predict(self, p: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        return expit(self.a * logit(p) + self.b)


@dataclass
class GroupShrunkPlatt:
    a: float
    b: float
    lam: float
    group_ids: np.ndarray  # sorted codes of the groups that got their own offsets
    da: np.ndarray
    db: np.ndarray
    n_nonpositive_slopes: int  # 0 by construction; kept as a sanity check
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
            grad = np.concatenate([np.bincount(gi, r * z, G) + 2 * lam * da, np.bincount(gi, r, G) + 2 * lam * db])
            return loss, grad

        # a + da_g > 0 is enforced with bounds so the mapping stays increasing in every group
        # (then within-group AUC cannot change); intercept offsets are unbounded.
        bounds = [(-base.a + MIN_SLOPE, None)] * G + [(None, None)] * G
        res = minimize(f, np.zeros(2 * G), jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000})
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
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1e8, solver="lbfgs", max_iter=1000, tol=1e-10).fit(cls._x(p, train_clicks), y)
        return cls(lr.coef_[0].copy(), float(lr.intercept_[0]))

    def predict(self, p: np.ndarray, train_clicks: np.ndarray) -> np.ndarray:
        return expit(self._x(p, train_clicks) @ self.w + self.w0)


def save_calibrators(
    glob: GlobalPlatt, gs: GroupShrunkPlatt, fa: FreqAware, group_clicks: pl.DataFrame, path: Path
) -> None:
    np.savez(path, a=glob.a, b=glob.b, lam=gs.lam, group_ids=gs.group_ids, da=gs.da, db=gs.db, fa_w=fa.w, fa_w0=fa.w0)
    group_clicks.write_parquet(path.with_suffix(".group_clicks.parquet"))


def load_calibrators(path: Path) -> tuple[GlobalPlatt, GroupShrunkPlatt, FreqAware, pl.DataFrame]:
    z = np.load(path.with_suffix(".npz"))
    glob = GlobalPlatt(float(z["a"]), float(z["b"]))
    gs = GroupShrunkPlatt(glob.a, glob.b, float(z["lam"]), z["group_ids"], z["da"], z["db"], 0, True)
    fa = FreqAware(z["fa_w"], float(z["fa_w0"]))
    return glob, gs, fa, pl.read_parquet(path.with_suffix(".group_clicks.parquet"))


# ------------------------------------------------------------------------------------------------ exact (iii) solver
def fit_group_shrunk_exact(
    p: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    base: GlobalPlatt,
    lam: float,
    tol: float = 1e-9,
    max_iter: int = 100,
) -> tuple[GroupShrunkPlatt, dict]:
    """Same objective as GroupShrunkPlatt.fit, solved exactly, one group at a time.

    With the global (a, b) and lam fixed, the objective splits into one convex two-parameter problem per group:
        sum_i softplus(s_i) - y_i s_i + lam (da^2 + db^2),   s_i = (a + da) z_i + (b + db),   a + da >= MIN_SLOPE.
    Each one gets Newton steps with a backtracking line search that stays inside the slope bound. Stops when the
    projected gradient (sum scale) is below `tol`.

    Why not the joint L-BFGS-B fit: it reported convergence, yet a tighter solve still moved the small-group LogLoss
    by up to 3.9e-6. On study C's selection data about 35% of the improvement it showed was optimizer error."""
    z = logit(p)
    gid, gi = np.unique(groups, return_inverse=True)
    order = np.argsort(gi, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(gi, minlength=gid.size))]
    da, db = np.zeros(gid.size), np.zeros(gid.size)
    worst, iters, at_bound = 0.0, 0, 0
    lo = -base.a + MIN_SLOPE

    def obj(zg, yg, x):
        s = (base.a + x[0]) * zg + (base.b + x[1])
        return float(np.logaddexp(0.0, s).sum() - (yg * s).sum() + lam * (x @ x))

    def grad_hess(zg, yg, x):
        q = expit((base.a + x[0]) * zg + (base.b + x[1]))
        r, w = q - yg, q * (1 - q)
        g = np.array([r @ zg + 2 * lam * x[0], r.sum() + 2 * lam * x[1]])
        H = np.array([[w @ (zg * zg) + 2 * lam, w @ zg], [w @ zg, w.sum() + 2 * lam]])
        return g, H

    def projected(x, g):
        on_bound = x[0] <= lo + 1e-15 and g[0] > 0  # the slope wants to go below the bound
        return (np.array([0.0, g[1]]) if on_bound else g), on_bound

    for k in range(gid.size):
        rows = order[bounds[k] : bounds[k + 1]]
        zg, yg = z[rows], y[rows].astype(float)
        # start inside the bound: if the global slope is below MIN_SLOPE, a zero offset is already infeasible
        x = np.array([max(0.0, lo), 0.0])
        f = obj(zg, yg, x)
        # `it` is read after the loop for the iteration count
        for it in range(max_iter):  # noqa: B007
            g, H = grad_hess(zg, yg, x)
            pg, on_bound = projected(x, g)
            if np.abs(pg).max() < tol:
                break
            if on_bound:
                step = np.array([0.0, -g[1] / H[1, 1]])
            else:
                step = -np.linalg.solve(H, g)
                if x[0] + step[0] < lo and x[0] - lo < 1e-12:
                    # at the bound but the slope should move up (g[0] < 0); the coupled Newton step still points
                    # outside, so it would be cut to length 0. A diagonally scaled gradient step is a descent
                    # direction that moves the slope inward.
                    step = -g / np.diag(H)
            t = 1.0
            if step[0] < 0 and x[0] + step[0] < lo:  # stay feasible: shorten the step to the bound (x[0] >= lo)
                t = (lo - x[0]) / step[0]
            stalled = False
            while True:
                x_new = x + t * step
                x_new[0] = max(x_new[0], lo)
                f_new = obj(zg, yg, x_new)
                # Armijo, or no worse than float noise of the objective near the optimum
                if np.isfinite(f_new) and (f_new <= f + 1e-4 * (g @ (x_new - x)) or f_new <= f + 1e-13 * (1 + abs(f))):
                    break
                t *= 0.5
                if t < 1e-12:
                    stalled = True
                    break
            if stalled:  # keep the last accepted point; the final check below reports it
                break
            x, f = x_new, f_new
        iters = max(iters, it + 1)
        g, _ = grad_hess(zg, yg, x)  # convergence is judged at the returned point
        pg, _ = projected(x, g)
        worst = max(worst, float(np.abs(pg).max()))
        at_bound += int(x[0] <= lo + 1e-15)
        da[k], db[k] = x
    below = int((base.a + da < MIN_SLOPE - 1e-12).sum())
    gs = GroupShrunkPlatt(
        base.a, base.b, lam, gid, da, db, int((base.a + da <= 0).sum()), bool(worst < tol and below == 0)
    )
    return gs, {
        "solver": "per-group Newton",
        "max_projected_gradient": worst,
        "max_iterations": iters,
        "groups_at_slope_bound": at_bound,
        "slopes_below_bound": below,
        "global_slope": base.a,
        "tol": tol,
    }


def fit_group_intercept_exact(
    p: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    base: GlobalPlatt,
    lam: float,
    tol: float = 1e-9,
    max_iter: int = 100,
) -> tuple[GroupShrunkPlatt, dict]:
    """(iii-0): only the group intercept moves,
        logit p' = a·logit p + (b + db_g),   penalty lam·db_g^2   (the slope stays the global a).
    One convex 1-parameter problem per group, solved by Newton's method with a backtracking line search; stops
    when |gradient| (sum scale) < tol at the returned point. Returned as a GroupShrunkPlatt with da = 0."""
    z = logit(p)
    gid, gi = np.unique(groups, return_inverse=True)
    order = np.argsort(gi, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(gi, minlength=gid.size))]
    db = np.zeros(gid.size)
    worst, iters = 0.0, 0

    def obj(s0, yg, x):
        s = s0 + x
        return float(np.logaddexp(0.0, s).sum() - (yg * s).sum() + lam * x * x)

    for k in range(gid.size):
        rows = order[bounds[k] : bounds[k + 1]]
        s0, yg = base.a * z[rows] + base.b, y[rows].astype(float)
        x, f = 0.0, obj(s0, yg, 0.0)
        # `it` is read after the loop for the iteration count
        for it in range(max_iter):  # noqa: B007
            q = expit(s0 + x)
            g = float((q - yg).sum() + 2 * lam * x)
            if abs(g) < tol:
                break
            h = float((q * (1 - q)).sum() + 2 * lam)
            step, t = -g / h, 1.0
            while True:
                x_new = x + t * step
                f_new = obj(s0, yg, x_new)
                if np.isfinite(f_new) and (f_new <= f + 1e-4 * g * (x_new - x) or f_new <= f + 1e-13 * (1 + abs(f))):
                    break
                t *= 0.5
                if t < 1e-12:
                    break
            if t < 1e-12:
                break
            x, f = x_new, f_new
        iters = max(iters, it + 1)
        worst = max(worst, abs(float((expit(s0 + x) - yg).sum() + 2 * lam * x)))
        db[k] = x
    gs = GroupShrunkPlatt(
        base.a, base.b, lam, gid, np.zeros(gid.size), db, int(base.a <= 0) * gid.size, bool(worst < tol)
    )
    return gs, {
        "solver": "per-group Newton, intercept only",
        "max_abs_gradient": worst,
        "max_iterations": iters,
        "tol": tol,
    }


def save_group_calibrators(glob: GlobalPlatt, gs: GroupShrunkPlatt, path: Path) -> None:
    """(ii) and (iii) of one base model in one .npz; the groups are the codes `gs` was fitted on."""
    np.savez(path.with_suffix(".npz"), a=glob.a, b=glob.b, lam=gs.lam, group_ids=gs.group_ids, da=gs.da, db=gs.db)


def load_group_calibrators(path: Path) -> tuple[GlobalPlatt, GroupShrunkPlatt]:
    z = np.load(path.with_suffix(".npz"))
    glob = GlobalPlatt(float(z["a"]), float(z["b"]))
    return glob, GroupShrunkPlatt(glob.a, glob.b, float(z["lam"]), z["group_ids"], z["da"], z["db"], 0, True)
