"""M3: a lightweight neural CVR model that REFERENCES the MAC BASE design (not a reproduction).

Inputs (identical for both variants, so the variants differ only in pooling):
- other categorical inputs (CAT_INPUTS without ad_feat1/ad_feat2), main vocabulary codes
- target ids ad_feat0/1/2 and the 20-step purchase sequence of the same three id types,
  encoded with a SEPARATE id vocabulary counted over training targets AND sequences
  (M1/M2 codes are untouched); PAD -> MISSING (masked), rare/unseen -> UNKNOWN
- the two per-step visual-similarity scores provided by the data (mm_feat0/1); MAC BASE uses
  SimTier instead, so this is a simplification
- the same summary history features as M1/M2 (NUM_INPUTS), standardized

pooling="attn":  softmax over valid steps of a target-conditioned score MLP([q, k, q-k, q*k])
pooling="mean":  uniform weights over valid steps (the special case of equal attention scores)

Run this module in a process that does not use LightGBM: on this machine PyTorch and LightGBM
ship separate OpenMP runtimes, and using both in one process segfaults or hangs
(see cvr/m3_worker.py). Input arrays are built in cvr/m3data.py, which does not import torch.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .m3data import M3Arrays
from .schema import MISSING, MM_COLS


class M3Net(nn.Module):
    def __init__(self, other_sizes: list[int], id_sizes: list[int], n_num: int, dim: int, pooling: str,
                 hidden: tuple[int, ...] = (256, 128), att_hidden: int = 64, sparse: bool = True):
        super().__init__()
        if pooling not in ("attn", "mean"):
            raise ValueError(pooling)
        self.pooling = pooling
        self.other_emb = nn.ModuleList([nn.Embedding(s, dim) for s in other_sizes])
        # id tables are shared by the target and the sequence; sparse gradients keep big tables cheap (CPU)
        self.id_emb = nn.ModuleList([nn.Embedding(s, dim, sparse=sparse) for s in id_sizes])
        q_dim = dim * len(id_sizes)
        self.key_proj = nn.Linear(q_dim + len(MM_COLS), q_dim)
        self.att = nn.Sequential(nn.Linear(4 * q_dim, att_hidden), nn.ReLU(), nn.Linear(att_hidden, 1))
        in_dim = dim * len(other_sizes) + 2 * q_dim + n_num
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.mlp = nn.Sequential(*layers, nn.Linear(prev, 1))

    def forward(self, other, target, seq, mm, num):
        q = torch.cat([emb(target[:, i]) for i, emb in enumerate(self.id_emb)], dim=-1)  # (B, q)
        k = torch.cat([emb(seq[..., i]) for i, emb in enumerate(self.id_emb)] + [mm], dim=-1)  # (B, L, q+2)
        k = self.key_proj(k)  # (B, L, q)
        valid = seq[..., 0] != MISSING  # PAD steps are masked
        if self.pooling == "attn":
            qe = q.unsqueeze(1).expand_as(k)
            score = self.att(torch.cat([qe, k, qe - k, qe * k], dim=-1)).squeeze(-1)
        else:
            score = torch.zeros(valid.shape, device=k.device)
        # finite fill (not -inf) keeps gradients finite; multiplying by the mask afterwards gives
        # exactly zero weight on PAD steps and a zero pooled vector when the whole history is PAD
        w = torch.softmax(score.masked_fill(~valid, -1e9), dim=1) * valid
        pooled = (w.unsqueeze(-1) * k).sum(dim=1)
        x = torch.cat([emb(other[:, i]) for i, emb in enumerate(self.other_emb)] + [q, pooled, num], dim=-1)
        return self.mlp(x).squeeze(-1)


# ---------------------------------------------------------------- training / prediction
@dataclass
class M3Model:
    net: M3Net
    num_mean: np.ndarray
    num_std: np.ndarray
    device: str
    info: dict

    def _batch(self, a: M3Arrays, idx: np.ndarray):
        d = self.device
        return (torch.as_tensor(a.other[idx], dtype=torch.long, device=d),
                torch.as_tensor(a.target[idx], dtype=torch.long, device=d),
                torch.as_tensor(a.seq[idx], dtype=torch.long, device=d),
                torch.as_tensor(a.mm[idx], device=d),
                torch.as_tensor((a.num[idx] - self.num_mean) / self.num_std, device=d))

    @classmethod
    def fit(cls, a: M3Arrays, y: np.ndarray, other_sizes: list[int], id_sizes: list[int], pooling: str,
            seed: int, dim: int = 16, epochs: int = 1, batch: int = 4096, lr: float = 1e-3,
            device: str = "cpu", threads: int = 8) -> "M3Model":
        torch.manual_seed(seed)
        torch.set_num_threads(threads)
        rng = np.random.default_rng(seed)
        sparse = device == "cpu"
        net = M3Net(other_sizes, id_sizes, a.num.shape[1], dim, pooling, sparse=sparse).to(device)
        mean, std = a.num.mean(0), a.num.std(0) + 1e-6
        model = cls(net, mean.astype(np.float32), std.astype(np.float32), device, {})
        id_p = [p for e in net.id_emb for p in e.parameters()]
        dense_p = [p for n_, p in net.named_parameters() if not n_.startswith("id_emb")]
        opt_s = torch.optim.SparseAdam(id_p, lr=lr) if sparse else torch.optim.Adam(id_p, lr=lr)
        opt_d = torch.optim.Adam(dense_p, lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        yt = y.astype(np.float32)
        losses = []
        net.train()
        for _ in range(epochs):
            order = rng.permutation(a.n)
            for start in range(0, a.n, batch):
                idx = order[start:start + batch]
                logits = net(*model._batch(a, idx))
                loss = loss_fn(logits, torch.as_tensor(yt[idx], device=device))
                opt_s.zero_grad()
                opt_d.zero_grad()
                loss.backward()
                opt_s.step()
                opt_d.step()
                losses.append(loss.detach())
        losses = [float(x) for x in losses]
        model.info = {"pooling": pooling, "seed": seed, "dim": dim, "epochs": epochs, "batch": batch, "lr": lr,
                      "device": device, "steps": len(losses),
                      "train_loss_last100_mean": float(np.mean(losses[-100:])),
                      "params": int(sum(p.numel() for p in net.parameters()))}
        return model

    @torch.no_grad()
    def predict(self, a: M3Arrays, batch: int = 16384) -> np.ndarray:
        self.net.eval()
        out = np.empty(a.n, dtype=np.float64)
        for start in range(0, a.n, batch):
            idx = np.arange(start, min(start + batch, a.n))
            out[idx] = torch.sigmoid(self.net(*self._batch(a, idx))).cpu().numpy()
        return out


# ---------------------------------------------------------------- save / load
def save_m3(m: M3Model, other_sizes: list[int], id_sizes: list[int], path: Path) -> None:
    torch.save(m.net.state_dict(), path.with_suffix(".pt"))
    np.savez(path.with_suffix(".npz"), num_mean=m.num_mean, num_std=m.num_std)
    path.with_suffix(".json").write_text(json.dumps({"other_sizes": other_sizes, "id_sizes": id_sizes,
                                                     "info": m.info}))


def load_m3(path: Path, device: str = "cpu") -> M3Model:
    meta = json.loads(path.with_suffix(".json").read_text())
    z = np.load(path.with_suffix(".npz"))
    info = meta["info"]
    net = M3Net(meta["other_sizes"], meta["id_sizes"], n_num=len(z["num_mean"]), dim=info["dim"],
                pooling=info["pooling"], sparse=False)
    net.load_state_dict(torch.load(path.with_suffix(".pt"), map_location=device))
    return M3Model(net.to(device).eval(), z["num_mean"], z["num_std"], device, info)
