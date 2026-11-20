"""Second-study inputs built out-of-fold by user (docs/DESIGN.md §13).

- aux_mta: prediction of an auxiliary LightGBM trained on (mta > 0), a denser attribution label.
- te_<key>: beta-binomial smoothed last-click CVR of the row's ad_feat1 / ad_feat2 group.
- logn_<key>: log(1 + training clicks of the group) (counts only, no labels).

Training rows get values from models / statistics fitted without the row's user (K user folds);
V1 / V2 / test rows get values from the model / statistics fitted on all training rows.
The auxiliary label itself is never an input.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from .models import GBDT, SmoothedRate
from .schema import CAT_INPUTS, GROUP_KEY, NUM_INPUTS, SHOP_KEY

AUX_INPUTS = ["aux_mta"]
TE_KEYS = [GROUP_KEY, SHOP_KEY]
TE_INPUTS = [f"te_{k}" for k in TE_KEYS] + [f"logn_{k}" for k in TE_KEYS]
AUX_LABEL = "label_mta"


def user_fold(df: pl.DataFrame, k: int) -> np.ndarray:
    """A user-level fold id from a digit of the key that the other splits do not use."""
    return ((df["user_key"] // 10_000) % k).to_numpy()


@dataclass
class StackingModels:
    """Full-training fits used for every non-training row (V1, V2, test, serving)."""

    aux: GBDT | None
    te: dict[str, SmoothedRate]
    counts: dict[str, pl.DataFrame]
    aux_col: str = "aux_mta"  # "aux_last" for the same-label control of docs/DESIGN.md §15.2

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        cols = []
        if self.aux is not None:
            cols.append(pl.Series(self.aux_col, self.aux.predict(df)))
        for k, m in self.te.items():
            cols.append(pl.Series(f"te_{k}", m.predict(df)))
            n = df.select(k).join(self.counts[k], on=k, how="left", maintain_order="left")["n"].fill_null(0)
            cols.append(pl.Series(f"logn_{k}", np.log1p(n.to_numpy().astype(np.float64))))
        return df.with_columns(cols)


def fit_stacking(
    train: pl.DataFrame,
    lgb_params: dict,
    aux_seed: int,
    folds: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    use_aux: bool,
    use_te: bool,
    aux_label: str = AUX_LABEL,
    aux_col: str = "aux_mta",
) -> tuple[pl.DataFrame, StackingModels, dict]:
    """Returns (train with out-of-fold extra columns, full-training models, info).

    aux_label / aux_col default to the second study's mta model. The control of docs/DESIGN.md §15.2 builds the
    same out-of-fold input from the main label ("y") under the column name "aux_last"."""
    fold = user_fold(train, folds)
    info: dict = {
        "folds": folds,
        "aux_label": aux_label if use_aux else None,
        "aux_col": aux_col,
        "aux_seed": aux_seed,
        "fold_rows": np.bincount(fold, minlength=folds).tolist(),
    }
    extra: dict[str, np.ndarray] = {}
    aux_full, te_full, counts = None, {}, {}
    if use_aux:
        oof = np.empty(train.height)
        rounds = []
        for f in range(folds):
            m = GBDT.fit(
                train.filter(pl.Series(fold != f)),
                lgb_params,
                aux_seed,
                num_boost_round,
                early_stopping_rounds,
                CAT_INPUTS + NUM_INPUTS,
                label=aux_label,
            )
            oof[fold == f] = m.predict(train.filter(pl.Series(fold == f)))
            rounds.append(m.rounds)
        extra[aux_col] = oof
        aux_full = GBDT.fit(
            train,
            lgb_params,
            aux_seed,
            num_boost_round,
            early_stopping_rounds,
            CAT_INPUTS + NUM_INPUTS,
            label=aux_label,
        )
        info["aux_rounds"] = {"folds": rounds, "full": aux_full.rounds}
        info["aux_label_rate"] = float(train[aux_label].mean())
    if use_te:
        for k in TE_KEYS:
            oof = np.empty(train.height)
            for f in range(folds):
                m = SmoothedRate.fit(train.filter(pl.Series(fold != f)), k)
                oof[fold == f] = m.predict(train.filter(pl.Series(fold == f)))
            extra[f"te_{k}"] = oof
            te_full[k] = SmoothedRate.fit(train, k)
            counts[k] = train.group_by(k).agg(pl.len().alias("n"))
            n = train.select(k).join(counts[k], on=k, how="left", maintain_order="left")["n"]
            extra[f"logn_{k}"] = np.log1p(n.to_numpy().astype(np.float64))  # counts only: no label leakage
        info["te_prior_strength"] = {k: m.m for k, m in te_full.items()}
    out = train.with_columns([pl.Series(c, v) for c, v in extra.items()])
    return out, StackingModels(aux_full, te_full, counts, aux_col), info
