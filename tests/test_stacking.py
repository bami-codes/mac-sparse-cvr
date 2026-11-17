"""Out-of-fold group-rate inputs must not use the row's own user (no LightGBM here: OpenMP, see m3_worker)."""

import numpy as np
import polars as pl

from cvr.stacking import fit_stacking, user_fold


def _frame():
    # user_key // 10_000 % 3 decides the fold: keys 0, 10_000, 20_000 -> folds 0, 1, 2
    users = np.repeat([0, 10_000, 20_000, 30_000, 40_000, 50_000], 50)
    g1 = np.where(users == 0, 7, 2)  # group 7 exists ONLY for the fold-0 user
    y = np.where(users == 0, 1, (np.arange(users.size) % 10 == 0)).astype(np.int8)  # group 7 always converts
    return pl.DataFrame({"user_key": users, "ad_feat1": g1.astype(np.int32), "ad_feat2": np.full(users.size, 3, np.int32),
                         "y": y, "label_mta": y})


def test_users_stay_in_one_fold():
    df = _frame()
    f = user_fold(df, 3)
    for u in np.unique(df["user_key"]):
        assert np.unique(f[df["user_key"].to_numpy() == u]).size == 1


def test_out_of_fold_group_rate_does_not_see_own_labels():
    df = _frame()
    out, models, info = fit_stacking(df, {}, 0, 3, 10, 5, use_aux=False, use_te=True)
    own = out.filter(pl.col("ad_feat1") == 7)
    # group 7 never appears outside fold 0, so its OOF rate is the out-of-fold prior, not ~1.0
    assert own["te_ad_feat1"].max() < 0.5
    # the full-training model (used for V1/V2/test rows) does see group 7's labels
    assert models.transform(own.head(1))["te_ad_feat1"][0] > 0.5
    # counts are label-free and equal to the full training count
    assert np.allclose(own["logn_ad_feat1"].to_numpy(), np.log1p(50))
