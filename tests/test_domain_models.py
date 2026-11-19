"""G-int offsets (docs/DESIGN.md §16.8 tests 4-5). No LightGBM here: this pytest process also loads PyTorch."""

import numpy as np
import polars as pl
from scipy.special import expit

from cvr.domain_models import G_INT_INPUTS, G_SHARED_INPUTS, GBDTOffset, early_stopping_offsets, oof_domain_offset
from cvr.models import SmoothedRate, early_stopping_users
from cvr.schema import DOMAIN_KEY
from cvr.stacking import user_fold


def _frame(seed=0, n_users=300, rows=20):
    rng = np.random.default_rng(seed)
    # keys spread over the digits used by the early-stopping rule (//100 % 10) and the folds (//10_000 % 3)
    keys = rng.choice(10**7, n_users, replace=False).astype(np.int64)
    users = np.repeat(keys, rows)
    dom = rng.integers(2, 6, users.size)
    y = (rng.random(users.size) < np.where(dom == 2, 0.2, 0.02)).astype(np.int8)
    return pl.DataFrame({"user_key": users, DOMAIN_KEY: dom.astype(np.int32), "y": y})


def test_g_int_has_no_domain_input_and_g_shared_has_it():
    assert DOMAIN_KEY not in G_INT_INPUTS and DOMAIN_KEY in G_SHARED_INPUTS
    assert set(G_SHARED_INPUTS) - set(G_INT_INPUTS) == {DOMAIN_KEY}


def test_oof_offset_does_not_see_the_rows_own_user():
    # domain 9 exists only for users of fold 0 and always converts there
    df = _frame()
    fold = user_fold(df, 3)
    only = pl.Series((fold == 0) & (np.arange(df.height) % 7 == 0))
    df = df.with_columns(pl.when(only).then(9).otherwise(pl.col(DOMAIN_KEY)).alias(DOMAIN_KEY),
                         pl.when(only).then(1).otherwise(pl.col("y")).cast(pl.Int8).alias("y"))
    off = oof_domain_offset(df)
    own = expit(off[only.to_numpy()])
    assert own.max() < 0.5  # fold 0 rows of domain 9 get the out-of-fold prior, not ~1
    full = SmoothedRate.fit(df, DOMAIN_KEY).predict(df.filter(only))
    assert full.min() > 0.5  # the all-rows rate (for V1/V2/test rows) does see those labels


def test_early_stopping_labels_never_reach_any_offset():
    df = _frame()
    es = early_stopping_users(df)
    assert 0.03 < es.mean() < 0.2
    flipped = df.with_columns(pl.when(pl.Series(es)).then(1 - pl.col("y")).otherwise(pl.col("y")).cast(pl.Int8).alias("y"))
    _, fit_a, es_a = early_stopping_offsets(df)
    _, fit_b, es_b = early_stopping_offsets(flipped)
    assert np.array_equal(fit_a, fit_b) and np.array_equal(es_a, es_b)
    # the wrong order (out-of-fold over all training rows, then drop the early-stopping rows) does leak them
    naive_a, naive_b = oof_domain_offset(df)[~es], oof_domain_offset(flipped)[~es]
    assert not np.array_equal(naive_a, naive_b)


class _FakeBooster:
    def predict(self, X, raw_score=False):
        if not raw_score:
            return np.full(X.shape[0], np.nan)  # the path that would ignore init_score must never be used
        return np.linspace(-1.0, 1.0, X.shape[0])


def test_prediction_adds_the_offset_exactly_once():
    df = _frame().with_columns([pl.lit(3).cast(pl.Int32).alias(c) for c in G_INT_INPUTS if c not in ("y",)])
    rate = SmoothedRate.fit(df, DOMAIN_KEY)
    m = GBDTOffset(_FakeBooster(), G_INT_INPUTS, 10, rate)
    raw = np.linspace(-1.0, 1.0, df.height)
    off = np.log(rate.predict(df) / (1 - rate.predict(df)))
    p = m.predict(df)
    assert np.allclose(p, expit(raw + off), rtol=0, atol=1e-12)
    assert not np.allclose(p, expit(raw + 2 * off)) and not np.allclose(p, expit(raw))
