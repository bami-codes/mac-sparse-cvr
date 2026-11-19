"""Study E serving (docs/DESIGN.md §18.8): raw MAC rows -> calibrated probability with an official PyMAL model, in
process on the CPU.

Path of one request: raw row dicts -> the official vocabularies (the same lookups as utils.data.df_to_dict) -> the
official tensor layout (the same as utils.data.MALDataset) -> model -> (ii) global Platt and (iii) group-shrunk Platt
on the row's ad_feat1 group (the frozen calibrators of study E). The two similarity sequences (mm_feat0_seq,
mm_feat1_seq) are precomputed in MAC and read as given: computing them is outside the timed path, as are network,
request parsing and feature-store lookups.

New inputs (review: a new user_id raised KeyError although the model never reads it):
- user_id is not looked up: the official model drops it (user_indices[:, 1:]); it only groups GAUC in evaluation;
- a value absent from an official vocabulary (a new ad, shop or history item) takes that vocabulary's PAD index, the
  missing value the model was trained with, and is counted in `oov` for monitoring;
- an ad_feat1 value absent from the training statistics (including any new value) is the pooled "unseen" group -1 of
  the frozen calibrators, whose coefficients were fitted on V1's unseen rows; a group the calibrator never saw at
  all takes the global (ii) mapping. This is the evaluated behavior, kept so that serving equals the evaluation.
"""

import argparse
from pathlib import Path

import numpy as np

from . import pymal_run as R
from . import study_e_eval as Q
from .domain_models import load_domain_calibrators
from .schema import PAD


class ServerE:
    def __init__(self, rid: str, ckpt: Path, vocabs: dict, codes: dict[str, int], threads: int = 1,
                 calibration: str = "iii"):
        """A study E model: the run record next to the checkpoint and the frozen calibrator of that run."""
        run = R.run_record(Path(ckpt))
        cal = Q.calibrator_path(rid, calibration, "main")
        self._setup(run["args"], Path(ckpt), vocabs, codes, cal, threads, {"run": rid})

    @classmethod
    def from_parts(cls, args: dict, ckpt: Path, vocabs: dict, codes: dict[str, int], calibrator: Path | None,
                   threads: int = 1, info: dict | None = None) -> "ServerE":
        """Any official model: its training arguments, state dict, vocabularies, calibrator group codes and file
        (calibrator None: a model without a calibrator fitted for it serves raw probabilities only)."""
        self = cls.__new__(cls)
        self._setup(args, Path(ckpt), vocabs, codes, Path(calibrator) if calibrator else None, threads, info or {})
        return self

    def _setup(self, args: dict, ckpt: Path, vocabs: dict, codes: dict[str, int], calibrator: Path | None,
               threads: int, info: dict) -> None:
        import torch
        torch.set_num_threads(threads)
        self.torch = torch
        self.D, _ = R.official()
        a = argparse.Namespace(**args)
        self.model, _, _ = R.build(a, vocabs, torch.device("cpu"))
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.eval()
        self.main_view = a.main_view
        self.vocabs, self.codes = vocabs, codes
        self.pad = {c: v[PAD] for c, v in vocabs.items()}  # the missing-value index of every official vocabulary
        self.oov: dict[str, int] = {}
        self.glob, self.gs = load_domain_calibrators(calibrator) if calibrator else (None, None)
        self.info = info | {"checkpoint": str(ckpt), "checkpoint_sha256": R.sha256(ckpt), "model": a.model,
                            "calibrator": calibrator.name if calibrator else None,
                            "lambda": float(self.gs.lam) if self.gs else None, "threads": threads}

    @staticmethod
    def columns() -> list[str]:
        D, _ = R.official()
        return D.USER_COLS + D.AD_COLS + D.SEQ_COLS + D.MM_COLS

    def code(self, col: str, value: str) -> int:
        i = self.vocabs[col].get(value)
        if i is None:
            self.oov[col] = self.oov.get(col, 0) + 1
            return self.pad[col]
        return i

    def user_part(self, row: dict):
        """User features and purchase history of one row: index 0 (user_id) is a placeholder, never read."""
        D, torch = self.D, self.torch
        user = [0] + [self.code(c, row[c]) for c in D.USER_COLS[1:]]
        seq = [[self.code(D.MAP_COLS[c], t) for t in row[c]] for c in D.SEQ_COLS]
        return torch.tensor([user], dtype=torch.long), torch.tensor([seq], dtype=torch.long)

    def ad_part(self, rows: list[dict]):
        D, torch = self.D, self.torch
        ad = torch.tensor([[self.code(c, r[c]) for c in D.AD_COLS] for r in rows], dtype=torch.long)
        mm = torch.tensor(np.asarray([[r[c] for c in D.MM_COLS] for r in rows], dtype=np.float32))
        return ad, mm

    def tensors(self, rows: list[dict]):
        """The official mapping of df_to_dict and the MALDataset layout, for a list of raw rows (each with its own user)."""
        parts = [self.user_part(r) for r in rows]
        ad, mm = self.ad_part(rows)
        return (self.torch.cat([u for u, _ in parts]), ad, mm, self.torch.cat([q for _, q in parts]))

    def request_tensors(self, user_row: dict, candidates: list[dict]):
        """One request: the user's features and history encoded ONCE and shared by every candidate (the per-row
        path looked the same history up once per candidate). The candidates bring their ad and context features and
        their precomputed similarity sequences."""
        user, seq = self.user_part(user_row)
        ad, mm = self.ad_part(candidates)
        n = len(candidates)
        return user.expand(n, -1), ad, mm, seq.expand(n, -1, -1)

    def score(self, tensors) -> np.ndarray:
        user, ad, mm, seq = tensors
        with self.torch.no_grad():
            p = self.model(user[:, 1:], ad, mm, seq)[self.main_view][:, 1]
        return p.numpy().astype(np.float64)

    def raw(self, rows: list[dict]) -> np.ndarray:
        return self.score(self.tensors(rows))

    def calibrate(self, p: np.ndarray, rows: list[dict]) -> np.ndarray:
        if self.gs is None:
            raise ValueError("this model has no calibrator fitted for it")
        g = np.array([self.codes.get(r["ad_feat1"], -1) for r in rows])  # -1: absent from the training statistics
        return self.gs.predict(p, g)

    def predict(self, rows: list[dict]) -> np.ndarray:
        """Calibrated probability (iii) of every row, each row with its own user."""
        return self.calibrate(self.raw(rows), rows)

    def predict_request(self, user_row: dict, candidates: list[dict]) -> np.ndarray:
        """Calibrated probability (iii) of every candidate of one user's request."""
        return self.calibrate(self.score(self.request_tensors(user_row, candidates)), candidates)
