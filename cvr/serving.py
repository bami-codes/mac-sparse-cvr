"""Serving path for latency measurement: one raw row (python dict of strings/lists) -> calibrated probability.

Everything a request needs happens here: string -> code mapping, purchase-history features,
model, and calibration. The training/evaluation path builds the same features with polars
(cvr/features.py); `scripts/latency.py` checks that both paths give identical features and
predictions before timing.

Library imports are lazy and per model type: an M3 server imports PyTorch but never LightGBM,
because the two OpenMP runtimes cannot share one process on this machine (cvr/m3_worker.py).
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import expit

from .calibration import load_calibrators
from .m3data import ID_COLS, OTHER_CATS, SEQ_LEN, M3Arrays
from .schema import CAT_INPUTS, GROUP_KEY, LR_ONEHOT, MM_COLS, NUM_INPUTS, PAD, SEQ_PAIRS, UNKNOWN, VOCAB_COLS


def _load_maps(folder: Path) -> dict[str, dict[str, int]]:
    return {p.stem: dict(zip(*pl.read_parquet(p).select("value", "code").to_dict(as_series=False).values()))
            for p in folder.glob("*.parquet")}


class Server:
    def __init__(self, run_dir: Path, base: str, method: str):
        self.base, self.method = base, method
        self.vocab = _load_maps(run_dir / "vocab")
        mdir = run_dir / "models"
        if base == "M1":
            from .artifacts import load_logreg

            self.model = load_logreg(mdir / "M1")
            self.coef = self.model.model.coef_[0]
            self.intercept = float(self.model.model.intercept_[0])
        elif base.startswith("M2"):
            from .artifacts import load_gbdt

            self.model = load_gbdt(mdir / base)
        elif base.startswith("M3"):
            import torch

            from .nn import load_m3

            torch.set_num_threads(1)
            self.model = load_m3(mdir / base)
            self.id_vocab = _load_maps(run_dir / "id_vocab")
        else:
            raise ValueError(base)
        self.glob, self.gs, self.fa, gc = load_calibrators(mdir / f"calib_{base}")
        self.group_clicks = dict(zip(gc[GROUP_KEY].to_list(), gc["tr_clicks"].to_list()))

    # ------------------------------------------------------------ features (must match cvr/features.py)
    def features(self, rows: list[dict]) -> tuple[dict[str, np.ndarray], np.ndarray]:
        codes = {c: np.fromiter((self.vocab[c].get(r[c], UNKNOWN) for r in rows), np.int64, len(rows))
                 for c in VOCAB_COLS}
        num = np.empty((len(rows), len(NUM_INPUTS)), dtype=np.float64)
        for i, r in enumerate(rows):
            vals = []
            for tgt, seq in SEQ_PAIRS:
                t = r[tgt]
                vals.append(0 if t == PAD else sum(1 for s in r[seq] if s == t))
            s0 = r["ad_feat0_seq"]
            vals.append(1 if all(s == PAD for s in s0) else 0)  # hist_empty
            for m in MM_COLS:
                a = np.asarray(r[m], dtype=np.float64)
                vals += [np.float32(a.max()), np.float32(a.mean())]  # float32 like the polars path
            num[i] = vals
        return codes, num

    # ------------------------------------------------------------ model scores
    def raw_score(self, rows: list[dict], codes: dict[str, np.ndarray], num: np.ndarray) -> np.ndarray:
        if self.base == "M1":
            m = self.model
            z = np.full(len(rows), self.intercept)
            for c in LR_ONEHOT:
                z += self.coef[codes[c] + m.offsets[c]]
            if m.num_inputs:
                z += ((num - m.mean) / m.std) @ self.coef[m.n_onehot:m.n_onehot + len(m.num_inputs)]
            return expit(z)
        if self.base.startswith("M2"):
            X = np.column_stack([codes[c] for c in CAT_INPUTS] + [num]).astype(np.float64)
            return self.model.booster.predict(X, num_threads=1)
        n = len(rows)
        tgt = np.empty((n, len(ID_COLS)), np.int32)
        seq = np.empty((n, SEQ_LEN, len(ID_COLS)), np.int32)
        for j, (t, s) in enumerate(SEQ_PAIRS):
            v = self.id_vocab[t]
            tgt[:, j] = [v.get(r[t], UNKNOWN) for r in rows]
            seq[:, :, j] = [[v.get(x, UNKNOWN) for x in r[s]] for r in rows]
        mm = np.stack([np.asarray([r[m] for r in rows], dtype=np.float32) for m in MM_COLS], axis=-1)
        other = np.column_stack([codes[c] for c in OTHER_CATS]).astype(np.int32)
        return self.model.predict(M3Arrays(other, tgt, seq, mm, num.astype(np.float32)), batch=n)

    def calibrate(self, p: np.ndarray, codes: dict[str, np.ndarray]) -> np.ndarray:
        g = codes[GROUP_KEY]
        if self.method == "ii":
            return self.glob.predict(p)
        if self.method == "iv":
            n = np.fromiter((self.group_clicks.get(int(x), 0) for x in g), np.float64, g.size)
            return self.fa.predict(p, n)
        return self.gs.predict(p, g)  # iii_lam*

    def predict(self, rows: list[dict]) -> np.ndarray:
        codes, num = self.features(rows)
        return self.calibrate(self.raw_score(rows, codes, num), codes)


def selection(run_dir: Path) -> dict:
    return json.loads((run_dir / "selection.json").read_text())
