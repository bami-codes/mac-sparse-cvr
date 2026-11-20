"""Serve an official PyMAL model on the CPU: raw MAC rows in, calibrated conversion probabilities out.

The lookups match utils.data.df_to_dict and the tensors match utils.data.MALDataset, so a served row gets the same
probability as in evaluation. The similarity sequences (mm_feat0_seq, mm_feat1_seq) come precomputed with MAC.

Inputs the model has not seen:
- user_id is never looked up. The official model drops that column (user_indices[:, 1:]).
- A value missing from a vocabulary maps to that vocabulary's PAD index, the missing value used in training, and is
  counted in `oov` so it can be monitored.
- An ad group missing from the training statistics uses the calibrator's pooled "unseen" group (-1), exactly as in
  evaluation.
"""

import argparse
from pathlib import Path

import numpy as np

from . import evaluation, pymal_run
from .calibration import load_group_calibrators
from .schema import PAD


class Server:
    def __init__(
        self, rid: str, ckpt: Path, vocabs: dict, codes: dict[str, int], threads: int = 1, calibration: str = "iii"
    ):
        """A model of the official code, with its run record next to the checkpoint and its calibrator."""
        run = pymal_run.run_record(Path(ckpt))
        cal = evaluation.calibrator_path(rid, calibration, "main")
        self._setup(run["args"], Path(ckpt), vocabs, codes, cal, threads, {"run": rid})

    @classmethod
    def from_parts(
        cls,
        args: dict,
        ckpt: Path,
        vocabs: dict,
        codes: dict[str, int],
        calibrator: Path | None,
        threads: int = 1,
        info: dict | None = None,
    ) -> "Server":
        """Any official model. With calibrator=None it can only return raw probabilities."""
        self = cls.__new__(cls)
        self._setup(args, Path(ckpt), vocabs, codes, Path(calibrator) if calibrator else None, threads, info or {})
        return self

    def _setup(
        self,
        args: dict,
        ckpt: Path,
        vocabs: dict,
        codes: dict[str, int],
        calibrator: Path | None,
        threads: int,
        info: dict,
    ) -> None:
        import torch

        torch.set_num_threads(threads)
        self.torch = torch
        self.D, _ = pymal_run.official()
        a = argparse.Namespace(**args)
        self.model, _, _ = pymal_run.build(a, vocabs, torch.device("cpu"))
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.eval()
        self.main_view = a.main_view
        self.vocabs, self.codes = vocabs, codes
        self.pad = {c: v[PAD] for c, v in vocabs.items()}
        self.oov: dict[str, int] = {}
        self.glob, self.gs = load_group_calibrators(calibrator) if calibrator else (None, None)
        self.info = info | {
            "checkpoint": str(ckpt),
            "checkpoint_sha256": pymal_run.sha256(ckpt),
            "model": a.model,
            "calibrator": calibrator.name if calibrator else None,
            "lambda": float(self.gs.lam) if self.gs else None,
            "threads": threads,
        }

    @staticmethod
    def columns() -> list[str]:
        D, _ = pymal_run.official()
        return D.USER_COLS + D.AD_COLS + D.SEQ_COLS + D.MM_COLS

    def code(self, col: str, value: str) -> int:
        i = self.vocabs[col].get(value)
        if i is None:
            self.oov[col] = self.oov.get(col, 0) + 1
            return self.pad[col]
        return i

    def user_part(self, row: dict):
        """User features and purchase history as tensors. Slot 0 stands in for user_id, which the model ignores."""
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
        """Tensors for rows that each carry their own user, in MALDataset's layout."""
        parts = [self.user_part(r) for r in rows]
        ad, mm = self.ad_part(rows)
        return (self.torch.cat([u for u, _ in parts]), ad, mm, self.torch.cat([q for _, q in parts]))

    def request_tensors(self, user_row: dict, candidates: list[dict]):
        """Tensors for one user's candidates. The user's history is looked up once and broadcast to every
        candidate instead of once per candidate, which removes most of the encoding time of a request."""
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
        g = np.array([self.codes.get(r["ad_feat1"], -1) for r in rows])  # -1: the pooled "unseen" group
        return self.gs.predict(p, g)

    def predict(self, rows: list[dict]) -> np.ndarray:
        """Calibrated probability per row, each row with its own user."""
        return self.calibrate(self.raw(rows), rows)

    def predict_request(self, user_row: dict, candidates: list[dict]) -> np.ndarray:
        """Calibrated probability per candidate of one user's request."""
        return self.calibrate(self.score(self.request_tensors(user_row, candidates)), candidates)
