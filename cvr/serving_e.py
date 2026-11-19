"""Study E serving (docs/DESIGN.md §18.8): raw MAC rows -> calibrated probability with an official PyMAL model, in
process on the CPU.

Path of one request: raw row dicts -> the official vocabularies (the same lookups as utils.data.df_to_dict) -> the
official tensor layout (the same as utils.data.MALDataset) -> model -> (ii) global Platt and (iii) group-shrunk Platt
on the row's ad_feat1 group (the frozen calibrators of study E). The two similarity sequences (mm_feat0_seq,
mm_feat1_seq) are precomputed in MAC and read as given: computing them is outside the timed path, as are network,
request parsing and feature-store lookups.
"""

import argparse
from pathlib import Path

import numpy as np

from . import pymal_run as R
from . import study_e_eval as Q
from .domain_models import load_domain_calibrators


class ServerE:
    def __init__(self, rid: str, ckpt: Path, vocabs: dict, codes: dict[str, int], threads: int = 1,
                 calibration: str = "iii"):
        import torch
        torch.set_num_threads(threads)
        self.torch = torch
        self.D, _ = R.official()
        run = R.run_record(Path(ckpt))
        a = argparse.Namespace(**run["args"])
        self.model, _, _ = R.build(a, vocabs, torch.device("cpu"))
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.eval()
        self.main_view = a.main_view
        self.vocabs, self.codes = vocabs, codes
        self.glob, self.gs = load_domain_calibrators(Q.calibrator_path(rid, calibration, "main"))
        self.info = {"run": rid, "checkpoint": str(ckpt), "checkpoint_sha256": R.sha256(Path(ckpt)), "model": a.model,
                     "calibrator": Q.calibrator_path(rid, calibration, "main").name, "lambda": float(self.gs.lam),
                     "threads": threads}

    @staticmethod
    def columns() -> list[str]:
        D, _ = R.official()
        return D.USER_COLS + D.AD_COLS + D.SEQ_COLS + D.MM_COLS

    def tensors(self, rows: list[dict]):
        """The official mapping of df_to_dict and the MALDataset layout, for a list of raw rows. An unknown value
        raises (the official vocabularies cover all of MAC; vocab.get would give None and fail later)."""
        D, v, torch = self.D, self.vocabs, self.torch
        user = torch.tensor([[v[c][r[c]] for c in D.USER_COLS] for r in rows], dtype=torch.long)
        ad = torch.tensor([[v[c][r[c]] for c in D.AD_COLS] for r in rows], dtype=torch.long)
        seq = torch.tensor([[[v[D.MAP_COLS[c]][t] for t in r[c]] for c in D.SEQ_COLS] for r in rows], dtype=torch.long)
        mm = torch.tensor(np.asarray([[r[c] for c in D.MM_COLS] for r in rows], dtype=np.float32))
        return user, ad, mm, seq

    def raw(self, rows: list[dict]) -> np.ndarray:
        user, ad, mm, seq = self.tensors(rows)
        with self.torch.no_grad():
            p = self.model(user[:, 1:], ad, mm, seq)[self.main_view][:, 1]
        return p.numpy().astype(np.float64)

    def predict(self, rows: list[dict]) -> np.ndarray:
        """Calibrated probability (iii) of every row; groups unseen in training take the global mapping."""
        p = self.raw(rows)
        g = np.array([self.codes.get(r["ad_feat1"], -1) for r in rows])
        return self.gs.predict(p, g)
