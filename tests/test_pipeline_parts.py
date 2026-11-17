"""Small synthetic checks for the parts reviewers asked about (no MAC data needed)."""

import numpy as np
import polars as pl
import pytest
from scipy.optimize import check_grad
from scipy.special import expit

from cvr import vocab as V
from cvr.calibration import GlobalPlatt, GroupShrunkPlatt, logit
from cvr.groups import classify
from cvr.metrics import paired_bootstrap_macro_diff
from cvr.schema import MISSING, PAD, UNKNOWN, assert_no_label_inputs


def test_vocab_is_shared_and_handles_unseen_and_pad():
    train = pl.DataFrame({"ad_feat1": ["b", "a", "b", PAD, "c", "b", "a"]})
    vocab = V.build_vocab(V.count_values([train], ["ad_feat1"]))
    # deterministic order: count desc, then value asc -> b (3) = 2, a (2) = 3, c (1) = 4
    mapping = dict(zip(vocab["ad_feat1"]["value"], vocab["ad_feat1"]["code"]))
    assert mapping == {PAD: MISSING, "b": 2, "a": 3, "c": 4}
    other = pl.DataFrame({"ad_feat1": ["c", "zzz", PAD, "a"]})  # a different split
    assert V.encode(other, vocab)["ad_feat1"].to_list() == [4, UNKNOWN, MISSING, 3]
    # rebuilding from the same data gives the same fingerprint
    assert V.fingerprint(vocab) == V.fingerprint(V.build_vocab(V.count_values([train], ["ad_feat1"])))


def test_label_guard():
    with pytest.raises(ValueError):
        assert_no_label_inputs(["user_feat0", "label_mta"])
    with pytest.raises(ValueError):
        assert_no_label_inputs(["last"])


def test_group_classes_use_training_stats_only():
    stats = pl.DataFrame({"ad_feat1": [2, 3, 4, MISSING], "tr_clicks": [600, 50, 900, 1000], "tr_pos": [5, 1, 80, 0]})
    df = pl.DataFrame({"ad_feat1": [2, 3, 4, MISSING, UNKNOWN, 9]})
    out = classify(df, stats, n_min=500, c_min=20, c_dense=50)
    assert out["group_class"].to_list() == ["G1", "small", "dense", "missing", "unseen", "unseen"]


def test_group_shrunk_platt_gradient_and_limits():
    rng = np.random.default_rng(0)
    n, G = 40000, 6  # ~500 positives per group, so the recovered shifts are precise
    g = rng.integers(G, size=n)
    p = rng.uniform(0.01, 0.2, size=n)
    true_shift = np.array([0.8, -0.5, 0.0, 0.3, -0.2, 0.0])
    y = (rng.uniform(size=n) < expit(logit(p) + true_shift[g])).astype(float)
    base = GlobalPlatt.fit(p, y)
    # huge penalty -> identical to the global mapping; tiny penalty -> group offsets move
    strong = GroupShrunkPlatt.fit(p, y, g, base, lam=1e9)
    assert np.allclose(strong.predict(p, g), base.predict(p), atol=1e-6)
    weak = GroupShrunkPlatt.fit(p, y, g, base, lam=1e-3)
    # slope and intercept offsets are collinear when logit(p) spans a narrow range, so compare
    # each group's average logit shift relative to the global mapping, not db alone
    shift = logit(weak.predict(p, g)) - logit(base.predict(p))
    mean_shift = np.array([shift[g == k].mean() for k in range(G)])
    assert np.corrcoef(mean_shift, true_shift)[0, 1] > 0.97
    # unseen group codes fall back to the global mapping
    assert np.allclose(weak.predict(p[:5], np.full(5, 99)), base.predict(p[:5]))

    # analytic gradient of the objective matches finite differences
    z = logit(p)
    gi = g

    def obj(theta, lam=2.0):
        da, db = theta[:G], theta[G:]
        s = (base.a + da[gi]) * z + (base.b + db[gi])
        return np.logaddexp(0, s).sum() - (y * s).sum() + lam * (da @ da + db @ db)

    def grad(theta, lam=2.0):
        da, db = theta[:G], theta[G:]
        s = (base.a + da[gi]) * z + (base.b + db[gi])
        r = expit(s) - y
        return np.concatenate([np.bincount(gi, r * z, G) + 2 * lam * da, np.bincount(gi, r, G) + 2 * lam * db])

    assert check_grad(obj, grad, rng.normal(scale=0.1, size=2 * G)) < 1e-3


def test_grouped_auc_matches_sklearn_and_gauc_weights_by_clicks():
    from sklearn.metrics import roc_auc_score

    from cvr.metrics import gauc, grouped_auc

    rng = np.random.default_rng(3)
    n = 3000
    g = rng.integers(15, size=n)
    p = np.round(rng.uniform(size=n), 2)  # rounding creates ties
    y = (rng.uniform(size=n) < p * 0.3).astype(float)
    y[g == 0] = 0  # one single-class group
    a = grouped_auc(y, p, g).sort("g")
    for row in a.iter_rows(named=True):
        m = g == row["g"]
        if row["auc"] is None:
            assert y[m].sum() in (0, m.sum())
        else:
            assert abs(row["auc"] - roc_auc_score(y[m], p[m])) < 1e-12
    out = gauc(y, p, g)
    used = a.filter(pl.col("auc").is_not_null())
    assert abs(out["gauc"] - (used["auc"] * used["n"]).sum() / used["n"].sum()) < 1e-12
    assert out["users_excluded"] >= 1


def test_seed_paired_bootstrap_is_mean_of_pairs():
    rng = np.random.default_rng(4)
    n = 3000
    y = (rng.uniform(size=n) < 0.05).astype(float)
    g, u = rng.integers(30, size=n), rng.integers(400, size=n)
    pa = [rng.uniform(0.01, 0.1, size=n) for _ in range(3)]
    pb = [rng.uniform(0.01, 0.1, size=n) for _ in range(3)]
    joint = paired_bootstrap_macro_diff(y, pa, pb, g, u, reps=20)
    singles = [paired_bootstrap_macro_diff(y, a, b, g, u, reps=20)["diff"] for a, b in zip(pa, pb)]
    assert abs(joint["diff"] - np.mean(singles)) < 1e-12
    assert np.allclose(joint["per_pair_diff"], singles)


def test_paired_bootstrap_identical_models_give_zero():
    rng = np.random.default_rng(1)
    n = 2000
    y = (rng.uniform(size=n) < 0.05).astype(float)
    p = rng.uniform(0.01, 0.1, size=n)
    out = paired_bootstrap_macro_diff(y, p, p, rng.integers(20, size=n), rng.integers(300, size=n), reps=50)
    assert out["diff"] == 0 and out["ci95"] == [0.0, 0.0]
