"""Exact per-group solver of the (iii) objective (study C, docs/DESIGN.md §16 revision 4)."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from cvr.calibration import MIN_SLOPE, GlobalPlatt, GroupShrunkPlatt, fit_group_shrunk_exact, logit


def _data(seed=0):
    rng = np.random.default_rng(seed)
    sizes = [20_000, 5_000, 800, 300, 60, 2_000]
    g = np.concatenate([np.full(n, k + 2) for k, n in enumerate(sizes)])
    z = rng.normal(-3.5, 1.0, g.size)
    shift = np.array([0.0, 0.4, -0.6, 0.9, 0.2, 0.0])[g - 2]
    slope = np.array([1.0, 0.7, 1.3, 1.0, 1.0, -0.8])[g - 2]  # group 7: anti-correlated -> slope bound binds
    y = (rng.random(g.size) < expit(slope * z + shift + (slope < 0) * -6.0)).astype(float)
    return expit(z), y, g


def _objective(p, y, g, base, lam, da, db, gid):
    z = logit(p)
    k = np.searchsorted(gid, g)
    s = (base.a + da[k]) * z + (base.b + db[k])
    return float(np.logaddexp(0.0, s).sum() - (y * s).sum() + lam * (da @ da + db @ db))


def _reference(p, y, g, base, lam):
    """Each group's 2-parameter problem with a very tight L-BFGS-B (independent of the Newton code)."""
    z = logit(p)
    gid = np.unique(g)
    da, db = np.zeros(gid.size), np.zeros(gid.size)
    for k, v in enumerate(gid):
        zg, yg = z[g == v], y[g == v]

        def f(x):
            s = (base.a + x[0]) * zg + (base.b + x[1])
            r = expit(s) - yg
            return (np.logaddexp(0.0, s).sum() - (yg * s).sum() + lam * (x @ x),
                    np.array([r @ zg + 2 * lam * x[0], r.sum() + 2 * lam * x[1]]))

        res = minimize(f, np.zeros(2), jac=True, method="L-BFGS-B", bounds=[(-base.a + MIN_SLOPE, None), (None, None)],
                       options={"maxiter": 100_000, "ftol": 1e-16, "gtol": 1e-12})
        da[k], db[k] = res.x
    return gid, da, db


def test_exact_solver_matches_a_tight_reference_and_beats_the_joint_fit():
    p, y, g = _data()
    base = GlobalPlatt.fit(p, y)
    for lam in (1.0, 100.0):
        exact, info = fit_group_shrunk_exact(p, y, g, base, lam)
        assert exact.converged and info["max_projected_gradient"] < 1e-9
        gid, da, db = _reference(p, y, g, base, lam)
        assert np.array_equal(gid, exact.group_ids)
        assert np.allclose(exact.da, da, atol=1e-5) and np.allclose(exact.db, db, atol=1e-5)
        f_exact = _objective(p, y, g, base, lam, exact.da, exact.db, gid)
        f_ref = _objective(p, y, g, base, lam, da, db, gid)
        joint = GroupShrunkPlatt.fit(p, y, g, base, lam)
        f_joint = _objective(p, y, g, base, lam, joint.da, joint.db, gid)
        assert f_exact <= f_ref + 1e-8 and f_exact <= f_joint + 1e-8


def test_slope_bound_binds_for_an_anti_correlated_group():
    p, y, g = _data()
    base = GlobalPlatt.fit(p, y)
    exact, info = fit_group_shrunk_exact(p, y, g, base, 1.0)
    k = int(np.flatnonzero(exact.group_ids == 7)[0])
    assert abs(base.a + exact.da[k] - MIN_SLOPE) < 1e-12
    assert info["groups_at_slope_bound"] >= 1
    assert (base.a + exact.da >= MIN_SLOPE - 1e-15).all()


def test_exact_calibrator_predicts_with_the_same_formula():
    p, y, g = _data()
    base = GlobalPlatt.fit(p, y)
    exact, _ = fit_group_shrunk_exact(p, y, g, base, 10.0)
    k = np.searchsorted(exact.group_ids, g)
    manual = expit((base.a + exact.da[k]) * logit(p) + (base.b + exact.db[k]))
    assert np.allclose(exact.predict(p, g), manual, rtol=0, atol=1e-15)
    unseen = exact.predict(p[:5], np.full(5, 99))
    assert np.allclose(unseen, base.predict(p[:5]))  # a group absent from the fit gets the global mapping


def test_global_slope_below_the_bound_still_gives_a_feasible_converged_fit():
    """A global slope of 0.000289 (< MIN_SLOPE) makes the zero start infeasible. The solver must still return a
    feasible fit, not keep that slope and report convergence, and must not divide by zero in the line search."""
    p, y, g = _data()
    base = GlobalPlatt(0.000288539, -4.0)
    for groups in (g, np.r_[g[:-1], 3]):  # the second: one row reassigned
        exact, info = fit_group_shrunk_exact(p, y, groups, base, 10.0)
        assert info["slopes_below_bound"] == 0 and (base.a + exact.da >= MIN_SLOPE - 1e-12).all()
        assert exact.converged and info["max_projected_gradient"] < 1e-9
        gid, da, db = _reference(p, y, groups, base, 10.0)
        f_exact = _objective(p, y, groups, base, 10.0, exact.da, exact.db, gid)
        f_ref = _objective(p, y, groups, base, 10.0, da, db, gid)
        assert f_exact <= f_ref + 1e-8
