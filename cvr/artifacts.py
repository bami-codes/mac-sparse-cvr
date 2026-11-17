"""Save / load M1 (logistic regression) and M2 (LightGBM) so predictions can be reproduced after reload
and the serving path (cvr/serving.py) can be timed. Files live under runs/<...>/models/ (never committed).

M3 lives in cvr/nn.py (save_m3 / load_m3) and calibrators in cvr/calibration.py, so that a PyTorch
process never has to import LightGBM (two OpenMP runtimes in one process crash on this machine).
"""

import json
from pathlib import Path

import numpy as np

from .models import GBDT, LogReg


def save_logreg(m: LogReg, path: Path) -> None:
    np.savez(path, coef=m.model.coef_, intercept=m.model.intercept_, mean=m.mean, std=m.std)
    path.with_suffix(".json").write_text(json.dumps({"offsets": m.offsets, "n_onehot": m.n_onehot,
                                                     "num_inputs": m.num_inputs}))


def load_logreg(path: Path) -> LogReg:
    from sklearn.linear_model import LogisticRegression

    z, meta = np.load(path.with_suffix(".npz")), json.loads(path.with_suffix(".json").read_text())
    lr = LogisticRegression()
    lr.coef_, lr.intercept_, lr.classes_ = z["coef"], z["intercept"], np.array([0, 1])
    return LogReg(meta["offsets"], meta["n_onehot"], meta["num_inputs"], z["mean"], z["std"], lr)


def save_gbdt(m: GBDT, path: Path) -> None:
    m.booster.save_model(str(path.with_suffix(".txt")))
    path.with_suffix(".json").write_text(json.dumps({"inputs": m.inputs, "rounds": m.rounds}))


def load_gbdt(path: Path) -> GBDT:
    import lightgbm as lgb

    meta = json.loads(path.with_suffix(".json").read_text())
    return GBDT(lgb.Booster(model_file=str(path.with_suffix(".txt"))), meta["inputs"], meta["rounds"])
