"""Study C evaluation: classes, metrics, weighted AUC, bootstrap, selection, power and verdict (synthetic)."""

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from cvr import domain_eval as D
from cvr import metrics as M
from cvr.schema import DOMAIN_KEY, UNKNOWN


def _stats():
    # 1,000,000 training rows: domain 2 large (60%), 3 large (1%), 4 small (0.5%), 5 tiny (0.2%), 6 tiny (2,999 rows)
    rows = {2: 600_000, 3: 10_000, 4: 5_000, 5: 2_000, 6: 2_999}
    rows[7] = 1_000_000 - sum(rows.values())  # 38.0% large
    df = pl.DataFrame({DOMAIN_KEY: np.concatenate([np.full(n, d) for d, n in rows.items()]).astype(np.int32),
                       "y": np.zeros(1_000_000, dtype=np.int8)})
    return D.domain_stats(df)


def test_domain_classes_use_training_share_and_clicks_only():
    st = _stats()
    probe = pl.DataFrame({DOMAIN_KEY: np.array([2, 3, 4, 5, 6, 7, 9, UNKNOWN], dtype=np.int32)})
    assert D.domain_class(probe, st).tolist() == ["large", "large", "small", "tiny", "tiny", "large", "unseen", "unseen"]
    s = D.domain_summary(st)
    assert s["small"]["codes"] == [4] and s["tiny"]["codes"] == [5, 6] and s["large"]["domains"] == 3


def test_weighted_auc_matches_sklearn_and_duplicated_rows():
    rng = np.random.default_rng(0)
    n = 3000
    d = rng.integers(0, 4, n)
    p = np.round(rng.random(n), 2)  # many ties
    y = (rng.random(n) < 0.2 + 0.3 * p).astype(int)
    idx = D.WeightedAUC(p, d, y, 4)
    a = idx.auc(np.ones(n))
    for k in range(4):
        m = d == k
        assert a[k] == pytest.approx(roc_auc_score(y[m], p[m]), abs=1e-12)
    w = rng.integers(0, 4, n).astype(float)
    rep = np.repeat(np.arange(n), w.astype(int))
    for k in range(4):
        m = d[rep] == k
        assert idx.auc(w)[k] == pytest.approx(roc_auc_score(y[rep][m], p[rep][m]), abs=1e-12)


def _eval_data(seed=0, n_users=400):
    rng = np.random.default_rng(seed)
    users = np.repeat(rng.choice(10**6, n_users, replace=False), 15)
    n = users.size
    dom = rng.choice([2, 3, 4, 5], n, p=[0.5, 0.2, 0.2, 0.1])
    cls = np.where(np.isin(dom, [2]), "large", "small").astype(object)
    base = 1 / (1 + np.exp(-(-2 + rng.normal(size=n))))
    y = (rng.random(n) < base).astype(np.int8)
    pa = [np.clip(base + rng.normal(0, 0.02, n), 0.01, 0.99) for _ in range(3)]
    pb = [np.clip(base + rng.normal(0, 0.05, n), 0.01, 0.99) for _ in range(3)]
    return y, pa, pb, dom, cls, users


def test_bootstrap_point_estimates_match_direct_computation():
    y, pa, pb, dom, cls, users = _eval_data()
    out = D.paired_bootstrap(y, pa, pb, dom, cls, users, reps=50, seed=1)
    small = cls == "small"
    per_pair = []
    for a, b in zip(pa, pb):
        ma = M.per_group(y[small], a[small], dom[small])["logloss"].mean()
        mb = M.per_group(y[small], b[small], dom[small])["logloss"].mean()
        per_pair.append(ma - mb)
    assert out["point"]["per_pair_small_macro"] == pytest.approx(per_pair, abs=1e-12)
    assert out["point"]["small_macro"] == pytest.approx(np.mean(per_pair), abs=1e-12)
    large = cls == "large"
    direct_large = np.mean([M.bce(y[large], a[large]).mean() - M.bce(y[large], b[large]).mean() for a, b in zip(pa, pb)])
    assert out["point"]["large_click"] == pytest.approx(direct_large, abs=1e-12)
    auc = np.mean([np.mean([roc_auc_score(y[small & (dom == k)], a[small & (dom == k)])
                            - roc_auc_score(y[small & (dom == k)], b[small & (dom == k)]) for k in (3, 4, 5)])
                   for a, b in zip(pa, pb)])
    assert out["point"]["small_auc_macro"] == pytest.approx(auc, abs=1e-12) and out["point"]["auc_domains"] == 3
    for k, (lo, hi) in out["ci95"].items():
        assert lo <= out["point"][k] <= hi, k


def test_bootstrap_of_identical_models_is_exactly_zero():
    y, pa, _, dom, cls, users = _eval_data()
    out = D.paired_bootstrap(y, pa, [a.copy() for a in pa], dom, cls, users, reps=20, seed=1)
    assert out["point"]["small_macro"] == 0 and out["ci95"]["small_macro"] == [0.0, 0.0]


def test_users_seen_in_both_test_files_are_one_cluster():
    y, pa, pb, dom, cls, users = _eval_data()
    two = D.paired_bootstrap(np.r_[y, y], [np.r_[a, a] for a in pa], [np.r_[b, b] for b in pb], np.r_[dom, dom],
                             np.r_[cls, cls], np.r_[users, users], reps=10, seed=1, with_auc=False)
    assert two["users"] == np.unique(users).size


def test_leave_one_domain_out_matches_manual_removal():
    y, pa, pb, dom, cls, users = _eval_data()
    out = D.leave_one_domain_out(y, pa, pb, dom, cls)
    small = cls == "small"
    for removed in (3, 4, 5):
        keep = small & (dom != removed)
        manual = np.mean([M.per_group(y[keep], a[keep], dom[keep])["logloss"].mean()
                          - M.per_group(y[keep], b[keep], dom[keep])["logloss"].mean() for a, b in zip(pa, pb)])
        assert out["by_removed_domain"][removed] == pytest.approx(manual, abs=1e-12)


def test_choose_candidate_tie_rule():
    assert D.choose_candidate({"N-star": 0.05 - 5e-7, "G-shared": 0.05, "N-head": 0.06}) == "G-shared"
    assert D.choose_candidate({"N-shared": 0.05, "N-head": 0.0499}) == "N-head"
    assert D.choose_candidate({"N-head": 0.05, "N-star": 0.05}) == "N-head"


def test_power_check_compares_in_absolute_logloss_units():
    out = D.power_check(h_v2=0.0002, l_int=0.05)
    assert out["h_test_optimistic"] == pytest.approx(0.0002 / np.sqrt(2))
    assert out["detectable_80pct"] == pytest.approx(1.4286 * 0.0002 / np.sqrt(2), rel=1e-3)
    assert out["adoption_effect_abs"] == pytest.approx(0.00015)
    assert out["underpowered_expected"] is True


def _boot(diff, hi, pairs, large_hi, seeds=(2026, 2027, 2028)):
    return {"point": {"small_macro": diff, "per_pair_small_macro": pairs, "per_pair_seeds": list(seeds)},
            "ci95": {"small_macro": [diff - 1e-4, hi], "large_click": [-1e-4, large_hi]}}


P99 = {2026: 0.2, 2027: 0.3, 2028: 0.25}


def test_verdict_names_each_failed_condition():
    v = D.verdict(_boot(-0.0002, -0.00005, [-1e-4, -2e-4, -3e-4], 5e-5), l_int=0.05, p99_by_seed=P99)
    assert v["tier"] == "adoption_candidate" and not v["failed_conditions"]
    v = D.verdict(_boot(-0.0002, -0.00005, [-1e-4, -2e-4, -3e-4], 5e-5), l_int=0.05, p99_by_seed=P99 | {2027: 1.2})
    assert v["tier"] == "improvement_confirmed" and v["failed_conditions"] == ["latency"]
    v = D.verdict(_boot(-0.0001, -0.00001, [-1e-4, -1e-4, -1e-4], 5e-5), l_int=0.05, p99_by_seed=P99)
    assert v["failed_conditions"] == ["effect_size"]  # 0.2% < 0.3%
    v = D.verdict(_boot(-0.0002, 0.00001, [-1e-4, -2e-4, -3e-4], 5e-5), l_int=0.05, p99_by_seed=P99)
    assert v["tier"] == "not_confirmed"
    v = D.verdict(_boot(-0.0002, -0.00005, [-1e-4, 1e-5, -3e-4], 5e-5), l_int=0.05, p99_by_seed=P99)
    assert v["tier"] == "not_confirmed"  # one seed pair positive
    v = D.verdict(_boot(-0.0002, -0.00005, [-1e-4, -2e-4, -3e-4], 5e-5), l_int=0.05, p99_by_seed=None)
    assert v["tier"] == "improvement_confirmed" and v["pending_conditions"] == ["latency"]


def test_verdict_refuses_missing_seed_evidence():
    ok = _boot(-0.0002, -0.00005, [-1e-4, -2e-4, -3e-4], 5e-5)
    one = _boot(-0.0002, -0.00005, [-1e-4], 5e-5, seeds=(2026,))
    none = _boot(-0.0002, -0.00005, [], 5e-5, seeds=())
    for b in (one, none):  # review: both gave adoption_candidate before
        with pytest.raises(ValueError, match="seed pairs"):
            D.verdict(b, l_int=0.05, p99_by_seed=P99)
    with pytest.raises(ValueError, match="p99 measured"):
        D.verdict(ok, l_int=0.05, p99_by_seed={2026: 0.2, 2027: 0.3})
    v = D.verdict(ok, l_int=0.05, p99_by_seed=P99 | {2028: 0.9})
    assert v["conditions"]["latency"]["p99_ms_max_over_seeds"] == 0.9


def test_bootstrap_records_the_seed_of_each_pair():
    y, pa, pb, dom, cls, users = _eval_data()
    out = D.paired_bootstrap(y, pa, pb, dom, cls, users, reps=5, seed=1, with_auc=False, pair_seeds=[2026, 2027, 2028])
    assert out["point"]["per_pair_seeds"] == [2026, 2027, 2028]
    with pytest.raises(ValueError, match="pair_seeds"):
        D.paired_bootstrap(y, pa, pb, dom, cls, users, reps=5, pair_seeds=[2026])


def _files(**over):
    d = {"mode": "files", "train": ["train-08.parquet", "train-09.parquet", "train-10.parquet"],
         "v1": ["train-11.parquet"], "v2": ["train-12.parquet"]}
    return d | over


def test_development_config_guard():
    D.check_development_sources(_files())
    bad = [_files(v2=["train-13.parquet"]), _files(train=["train-14.parquet"]), _files(v1=["train-12.parquet"]),
           _files(test="test.parquet"), _files(v2=["../test/test.parquet"])]
    for d in bad:  # review: the first three passed before (only a `test` key was refused)
        with pytest.raises(ValueError):
            D.check_development_sources(d)
    smoke = {"mode": "smoke", "smoke_file": "train-08.parquet", "smoke_train_buckets": [0, 60],
             "smoke_v1_buckets": [60, 80], "smoke_v2_buckets": [80, 100]}
    D.check_development_sources(smoke)
    with pytest.raises(ValueError, match="holdout"):
        D.check_development_sources(smoke | {"smoke_file": "train-13.parquet"})
    with pytest.raises(ValueError, match="overlap"):
        D.check_development_sources(smoke | {"smoke_v1_buckets": [50, 80]})


def test_verdict_uses_the_preregistered_seeds_and_finite_latencies():
    one = _boot(-0.0002, -0.00005, [-1e-4], 5e-5, seeds=(2026,))
    with pytest.raises(ValueError, match="seed pairs"):  # review: a [2026]-only run gave adoption_candidate
        D.verdict(one, l_int=0.05, p99_by_seed={2026: 0.2})
    ok = _boot(-0.0002, -0.00005, [-1e-4, -2e-4, -3e-4], 5e-5)
    for bad in ({2026: 0.3, 2027: float("nan"), 2028: 0.4}, {2026: 0.3, 2027: 0.0, 2028: 0.4},
                {2026: 0.3, 2027: float("inf"), 2028: 0.4}):  # review: NaN gave adoption_candidate in one order
        with pytest.raises(ValueError, match="finite and positive"):
            D.verdict(ok, l_int=0.05, p99_by_seed=bad)


def test_development_config_guard_compares_resolved_paths(tmp_path):
    base = {"mode": "files", "root": str(tmp_path / "train"), "train": ["train-08.parquet"]}
    for v1, v2 in (("train-11.parquet", "./train-11.parquet"), ("train-11.parquet", "../train/train-11.parquet")):
        with pytest.raises(ValueError, match="listed for both"):  # review: "./train-11.parquet" passed before
            D.check_development_sources(base | {"v1": [v1], "v2": [v2]})
    with pytest.raises(ValueError, match="holdout"):
        D.check_development_sources(base | {"v1": ["train-11.parquet"], "v2": ["./sub/../train-13.parquet"]})


def test_development_config_guard_ignores_case(tmp_path):
    root = tmp_path / "train"
    root.mkdir()
    (root / "train-11.parquet").write_bytes(b"")
    base = {"mode": "files", "root": str(root), "train": ["train-08.parquet"]}
    with pytest.raises(ValueError, match="listed for both"):  # review: passed as two files on this Mac
        D.check_development_sources(base | {"v1": ["train-11.parquet"], "v2": ["TRAIN-11.PARQUET"]})
    with pytest.raises(ValueError, match="holdout"):
        D.check_development_sources(base | {"v1": ["train-11.parquet"], "v2": ["TRAIN-13.PARQUET"]})
