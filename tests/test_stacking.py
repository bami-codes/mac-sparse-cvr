"""Out-of-fold group-rate inputs must not use the row's own user (no LightGBM here: OpenMP, see m3_worker)."""

import numpy as np
import polars as pl

from cvr.stacking import fit_stacking, user_fold


def _frame():
    # user_key // 10_000 % 3 decides the fold: keys 0, 10_000, 20_000 -> folds 0, 1, 2
    users = np.repeat([0, 10_000, 20_000, 30_000, 40_000, 50_000], 50)
    g1 = np.where(users == 0, 7, 2)  # group 7 exists only for the fold-0 user
    y = np.where(users == 0, 1, (np.arange(users.size) % 10 == 0)).astype(np.int8)  # group 7 always converts
    return pl.DataFrame(
        {
            "user_key": users,
            "ad_feat1": g1.astype(np.int32),
            "ad_feat2": np.full(users.size, 3, np.int32),
            "y": y,
            "label_mta": y,
        }
    )


def test_users_stay_in_one_fold():
    df = _frame()
    f = user_fold(df, 3)
    for u in np.unique(df["user_key"]):
        assert np.unique(f[df["user_key"].to_numpy() == u]).size == 1


def test_out_of_fold_group_rate_does_not_see_own_labels():
    df = _frame()
    out, models, _info = fit_stacking(df, {}, 0, 3, 10, 5, use_aux=False, use_te=True)
    own = out.filter(pl.col("ad_feat1") == 7)
    # group 7 never appears outside fold 0, so its OOF rate is the out-of-fold prior, not ~1.0
    assert own["te_ad_feat1"].max() < 0.5
    # the full-training model (used for V1/V2/test rows) does see group 7's labels
    assert models.transform(own.head(1))["te_ad_feat1"][0] > 0.5
    # counts are label-free and equal to the full training count
    assert np.allclose(own["logn_ad_feat1"].to_numpy(), np.log1p(50))


class _FakeAux:
    def predict(self, df):
        return np.full(df.height, 0.25)


def test_control_input_keeps_its_own_column_name(tmp_path):
    # docs/DESIGN.md §15.2: the same-label control must not be written into the mta column
    import json

    from cvr.artifacts import load_stacking
    from cvr.stacking import StackingModels

    df = _frame()
    out = StackingModels(_FakeAux(), {}, {}, "aux_last").transform(df)
    assert "aux_last" in out.columns and "aux_mta" not in out.columns
    assert StackingModels(_FakeAux(), {}, {}).transform(df).columns[-1] == "aux_mta"  # default unchanged
    # runs saved before the control have no aux_col in meta.json: they still load as the mta input
    (tmp_path / "meta.json").write_text(json.dumps({"aux": False, "te": {}}))
    assert load_stacking(tmp_path).aux_col == "aux_mta"


def test_group_rate_fit_does_not_depend_on_row_order():
    # the fitted prior strength moved in its 10th digit between two runs on the same rows (group_by order)
    from cvr.models import SmoothedRate

    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "ad_feat1": rng.integers(0, 2_000, 50_000).astype(np.int32),
            "y": (rng.uniform(size=50_000) < 0.03).astype(np.int8),
        }
    )
    a = SmoothedRate.fit(df, "ad_feat1")
    for seed in range(3):
        b = SmoothedRate.fit(df.sample(fraction=1.0, shuffle=True, seed=seed), "ad_feat1")
        assert b.m == a.m and b.table.equals(a.table)
