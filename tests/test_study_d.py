"""Study D (docs/DESIGN.md §17): intercept-only group calibration and the group bootstrap (synthetic)."""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import expit

from cvr import domain_eval as D
from cvr import metrics as M
from cvr.calibration import GlobalPlatt, fit_group_intercept_exact, logit


def _data(seed=0):
    rng = np.random.default_rng(seed)
    sizes = [20_000, 5_000, 800, 300, 60]
    g = np.concatenate([np.full(n, k + 2) for k, n in enumerate(sizes)])
    z = rng.normal(-3.5, 1.0, g.size)
    shift = np.array([0.0, 0.4, -0.6, 0.9, 0.2])[g - 2]
    y = (rng.random(g.size) < expit(z + shift)).astype(float)
    return expit(z), y, g


def test_intercept_only_solver_matches_a_one_dimensional_reference():
    p, y, g = _data()
    base = GlobalPlatt.fit(p, y)
    for lam in (1.0, 100.0, 10000.0):
        gs, info = fit_group_intercept_exact(p, y, g, base, lam)
        assert gs.converged and info["max_abs_gradient"] < 1e-9 and np.all(gs.da == 0)
        z = logit(p)
        for k, v in enumerate(gs.group_ids):
            s0, yg = base.a * z[g == v] + base.b, y[g == v]
            ref = minimize_scalar(lambda x: np.logaddexp(0.0, s0 + x).sum() - (yg * (s0 + x)).sum() + lam * x * x,
                                  bracket=(-3, 3), tol=1e-14).x
            assert gs.db[k] == pytest.approx(ref, abs=1e-6)
        k = np.searchsorted(gs.group_ids, g)
        assert np.allclose(gs.predict(p, g), expit(base.a * z + base.b + gs.db[k]), rtol=0, atol=1e-15)


def test_intercept_only_keeps_the_within_group_ranking():
    p, y, g = _data()
    base = GlobalPlatt.fit(p, y)
    gs, _ = fit_group_intercept_exact(p, y, g, base, 1.0)
    q = gs.predict(p, g)
    for v in np.unique(g):
        m = g == v
        assert np.array_equal(np.argsort(p[m], kind="stable"), np.argsort(q[m], kind="stable"))


def _eval(seed=1):
    rng = np.random.default_rng(seed)
    users = np.repeat(rng.choice(10**6, 500, replace=False), 12)
    n = users.size
    g = rng.choice([2, 3, 4, 5], n)
    base = expit(rng.normal(-2.5, 0.8, n))
    y = (rng.random(n) < base).astype(np.int8)
    pa = [np.clip(base * rng.uniform(0.9, 1.1), 1e-4, 0.99) for _ in range(3)]
    pb = [np.clip(base * 1.2, 1e-4, 0.99) for _ in range(3)]
    return y, pa, pb, g, users


def test_group_bootstrap_points_match_direct_computation():
    y, pa, pb, g, users = _eval()
    out = D.paired_bootstrap_groups(y, pa, pb, g, users, reps=30, seed=2, pair_seeds=[2026, 2027, 2028])
    macro, click, pois = [], [], []
    for a, b in zip(pa, pb):
        ga, gb = M.per_group(y, a, g), M.per_group(y, b, g)
        macro.append(ga["logloss"].mean() - gb["logloss"].mean())
        click.append(M.bce(y, a).mean() - M.bce(y, b).mean())
        pois.append(D.poisson_deviance(ga["O"], ga["E"]).mean() - D.poisson_deviance(gb["O"], gb["E"]).mean())
    assert out["point"]["macro"] == pytest.approx(np.mean(macro), abs=1e-12)
    assert out["point"]["click"] == pytest.approx(np.mean(click), abs=1e-12)
    assert out["point"]["poisson"] == pytest.approx(np.mean(pois), abs=1e-10)
    assert out["per_pair"]["poisson"] == pytest.approx(pois, abs=1e-10)
    for k, (lo, hi) in out["ci95"].items():
        assert lo <= out["point"][k] <= hi, k


def test_group_bootstrap_of_identical_models_is_zero():
    y, pa, _, g, users = _eval()
    out = D.paired_bootstrap_groups(y, pa, [a.copy() for a in pa], g, users, reps=10, seed=2)
    assert out["point"] == {"macro": 0.0, "click": 0.0, "poisson": 0.0}
    assert all(v == [0.0, 0.0] for v in out["ci95"].values())
