"""Study C GBDT family (docs/DESIGN.md §16.4).

G-shared: the M2 GBDT (cvr/models.GBDT) on CAT_INPUTS + NUM_INPUTS, so the domain is a categorical input.
G-int:    a GBDT without the domain input, boosted from a per-domain logit offset (LightGBM init_score):
          prediction = sigmoid(tree raw score + offset), the offset added exactly once.

Offsets are logit(beta-binomial smoothed domain rate). Fitting order, chosen so no row's offset is computed from its own label:
1. split off the early-stopping users (the same ~10% rule as GBDT.fit);
2. on the remaining fitting side only, user 3-fold out-of-fold offsets for the fitting rows;
3. early-stopping rows get the offset fitted on the whole fitting side;
4. once the rounds are fixed, refit on all training rows with user 3-fold out-of-fold offsets over all rows.
V1 / V2 / test / serving rows get the offset fitted on all training rows.

The offset logic below needs no LightGBM (tests run in the same process as PyTorch; OpenMP, see m3_worker).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.special import expit

from .calibration import logit
from .models import SmoothedRate, early_stopping_users
from .schema import CAT_INPUTS, DOMAIN_KEY, NUM_INPUTS, assert_no_label_inputs
from .stacking import user_fold

G_SHARED_INPUTS = CAT_INPUTS + NUM_INPUTS
G_INT_INPUTS = [c for c in CAT_INPUTS if c != DOMAIN_KEY] + NUM_INPUTS
assert_no_label_inputs(G_SHARED_INPUTS + G_INT_INPUTS)
OFFSET_FOLDS = 3


def oof_domain_offset(df: pl.DataFrame, folds: int = OFFSET_FOLDS) -> np.ndarray:
    """Per-row logit domain rate fitted without the row's user (user folds of cvr/stacking.user_fold)."""
    fold = user_fold(df, folds)
    out = np.empty(df.height)
    for f in range(folds):
        rate = SmoothedRate.fit(df.filter(pl.Series(fold != f)), DOMAIN_KEY)
        out[fold == f] = logit(rate.predict(df.filter(pl.Series(fold == f))))
    return out


def early_stopping_offsets(train: pl.DataFrame, folds: int = OFFSET_FOLDS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(early-stopping mask, offsets of the fitting rows, offsets of the early-stopping rows). The early-stopping
    users' labels enter neither (steps 1-3 of the module docstring)."""
    es = early_stopping_users(train)
    fit_side, es_side = train.filter(pl.Series(~es)), train.filter(pl.Series(es))
    off_fit = oof_domain_offset(fit_side, folds)
    off_es = logit(SmoothedRate.fit(fit_side, DOMAIN_KEY).predict(es_side))
    return es, off_fit, off_es


@dataclass
class GBDTOffset:
    booster: Any  # lightgbm.Booster, or any object with predict(X, raw_score=True)
    inputs: list[str]
    rounds: int
    rate: SmoothedRate  # fitted on all training rows; gives the offset of every non-training row

    def offset(self, df: pl.DataFrame) -> np.ndarray:
        return logit(self.rate.predict(df))

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        raw = self.booster.predict(df.select(self.inputs).to_numpy().astype(np.float64), raw_score=True)
        return expit(raw + self.offset(df))

    @classmethod
    def fit(cls, train: pl.DataFrame, params: dict, seed: int, num_boost_round: int, early_stopping_rounds: int,
            folds: int = OFFSET_FOLDS) -> "GBDTOffset":
        import lightgbm as lgb

        inputs = G_INT_INPUTS
        cat_idx = [i for i, c in enumerate(inputs) if c in CAT_INPUTS]
        X, y = train.select(inputs).to_numpy().astype(np.float64), train["y"].to_numpy()
        p = dict(params, seed=seed, objective="binary", deterministic=True, force_row_wise=True, verbose=-1)
        es, off_fit, off_es = early_stopping_offsets(train, folds)
        dtr = lgb.Dataset(X[~es], y[~es], init_score=off_fit, categorical_feature=cat_idx, free_raw_data=True)
        des = lgb.Dataset(X[es], y[es], init_score=off_es, categorical_feature=cat_idx, reference=dtr)
        probe = lgb.train(p, dtr, num_boost_round=num_boost_round, valid_sets=[des],
                          callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
        rounds = max(1, probe.best_iteration)
        full = lgb.train(p, lgb.Dataset(X, y, init_score=oof_domain_offset(train, folds), categorical_feature=cat_idx,
                                        free_raw_data=True), num_boost_round=rounds)
        return cls(full, inputs, rounds, SmoothedRate.fit(train, DOMAIN_KEY))


# ---------------------------------------------------------------- save / load (runs/<...>/models/, never committed)
def save_gbdt_offset(m: GBDTOffset, path: Path) -> None:
    m.booster.save_model(str(path.with_suffix(".txt")))
    m.rate.table.write_parquet(path.with_suffix(".rate.parquet"))
    path.with_suffix(".json").write_text(json.dumps({"inputs": m.inputs, "rounds": m.rounds, "rate_key": m.rate.key,
                                                     "rate_p0": m.rate.p0, "rate_m": m.rate.m}))


def load_gbdt_offset(path: Path) -> GBDTOffset:
    import lightgbm as lgb

    meta = json.loads(path.with_suffix(".json").read_text())
    rate = SmoothedRate(meta["rate_key"], meta["rate_p0"], meta["rate_m"], pl.read_parquet(path.with_suffix(".rate.parquet")))
    return GBDTOffset(lgb.Booster(model_file=str(path.with_suffix(".txt"))), meta["inputs"], meta["rounds"], rate)


def save_domain_calibrators(glob, gs, path: Path) -> None:
    """(ii) and (iii-d) of one base model (cvr/calibration.py), groups = domain codes."""
    np.savez(path.with_suffix(".npz"), a=glob.a, b=glob.b, lam=gs.lam, group_ids=gs.group_ids, da=gs.da, db=gs.db)


def load_domain_calibrators(path: Path):
    from .calibration import GlobalPlatt, GroupShrunkPlatt

    z = np.load(path.with_suffix(".npz"))
    glob = GlobalPlatt(float(z["a"]), float(z["b"]))
    return glob, GroupShrunkPlatt(glob.a, glob.b, float(z["lam"]), z["group_ids"], z["da"], z["db"], 0, True)
