"""Study C serving path (docs/DESIGN.md §16.8): one raw row (dict of strings / lists) -> calibrated probability.

raw strings -> vocabulary codes and purchase-history features (cvr/serving.row_features, shared with the first
study) -> base model (G-int / G-shared / N-int / N-shared / N-head / N-star) -> (iii-d) group calibration keyed by
the DOMAIN (context_feat1) code. The first-study server calibrates by ad_feat1; using that key here would silently
serve a different model (0.021 max difference on a smoke run), so the key is DOMAIN_KEY by construction and
experiments/scripts/latency_c.py compares every served probability with an independent reference before timing.

Library imports are lazy and per family: a network server imports PyTorch, a GBDT server LightGBM, never both.
"""

import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from cvr.calibration import load_group_calibrators, logit
from cvr.schema import CAT_INPUTS, DOMAIN_KEY, NUM_INPUTS

from .serving import _load_maps, row_features

GBDT_FAMILIES = ("G-int", "G-shared")
NN_FAMILIES = ("N-int", "N-shared", "N-head", "N-star", "N-star-noPN")


def model_path(run_dir: Path, family: str, seed: int) -> Path:
    """Saved base model of one family and seed (networks carry the chosen epoch in the name)."""
    mdir = run_dir / "models"
    if family in GBDT_FAMILIES:
        return mdir / f"{family}_s{seed}"
    hits = sorted(mdir.glob(f"{family}_s{seed}_e*.json"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one saved {family}_s{seed}_e* model, found {hits}")
    return hits[0].with_suffix("")


class ServerC:
    def __init__(self, run_dir: Path, family: str, seed: int, calibration: str = "iiid"):
        self.family, self.seed, self.calibration = family, seed, calibration
        self.vocab = _load_maps(run_dir / "vocab")
        path = model_path(run_dir, family, seed)
        if family == "G-shared":
            from .artifacts import load_gbdt

            self.model = load_gbdt(path)
        elif family == "G-int":
            from .domain_models import load_gbdt_offset

            self.model = load_gbdt_offset(path)
            t = self.model.rate.table
            self.rate = dict(zip(t[DOMAIN_KEY].to_list(), t["rate"].to_list()))
            self.rate_p0 = self.model.rate.p0
        elif family in NN_FAMILIES:
            import torch

            from .mdnet import load_md_model

            torch.set_num_threads(1)
            self.model = load_md_model(path)
        else:
            raise ValueError(family)
        self.glob, self.gs = load_group_calibrators(run_dir / "models" / f"calib_{family}_s{seed}")
        self.info = {
            "model": str(path),
            "calibrator": f"calib_{family}_s{seed}",
            "calibration": calibration,
            "lambda": self.gs.lam,
            "group_key": DOMAIN_KEY,
        }

    def features(self, rows: list[dict]) -> tuple[dict[str, np.ndarray], np.ndarray]:
        return row_features(self.vocab, rows)

    def raw_score(self, codes: dict[str, np.ndarray], num: np.ndarray) -> np.ndarray:
        n = num.shape[0]
        if self.family in GBDT_FAMILIES:
            cols = {c: codes[c].astype(np.float64) for c in CAT_INPUTS} | {
                c: num[:, j] for j, c in enumerate(NUM_INPUTS)
            }
            X = np.column_stack([cols[c] for c in self.model.inputs])
            if self.family == "G-shared":
                return self.model.booster.predict(X, num_threads=1)
            rate = np.fromiter((self.rate.get(int(x), self.rate_p0) for x in codes[DOMAIN_KEY]), np.float64, n)
            return expit(self.model.booster.predict(X, raw_score=True, num_threads=1) + logit(rate))
        from .mddata import MDArrays

        a = MDArrays(np.column_stack([codes[c] for c in CAT_INPUTS]).astype(np.int32), num.astype(np.float32))
        return self.model.predict(a, batch=n)

    def calibrate(self, p: np.ndarray, codes: dict[str, np.ndarray]) -> np.ndarray:
        if self.calibration == "ii":
            return self.glob.predict(p)
        return self.gs.predict(p, codes[DOMAIN_KEY])

    def predict(self, rows: list[dict]) -> np.ndarray:
        codes, num = self.features(rows)
        return self.calibrate(self.raw_score(codes, num), codes)


def selection(run_dir: Path) -> dict:
    return json.loads((run_dir / "selection.json").read_text())
