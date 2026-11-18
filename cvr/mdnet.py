"""Study C networks (docs/DESIGN.md §16.4): the M2 inputs, five ways of using the domain.

Domain = the DOMAIN_KEY (context_feat1) code. Inputs: CAT_INPUTS codes (embedded) + standardized NUM_INPUTS.

variant     how the domain is used                                                    mini-batches
int         no domain input; a per-domain logit intercept                             mixed
shared      the domain embedding is one input of a shared MLP                         mixed
head        shared body (domain is an input) + output (w + w_d)·h + (b + b_d)          mixed
star        STAR (Sheng et al., CIKM 2021): partitioned normalization (PN) on the     one domain per batch
            concatenated input, star-topology FCN in every layer (W ⊙ W_d, b + b_d),
            auxiliary 2-layer network on the domain embedding added to the logit
star_nopn   star FCN + auxiliary network, no normalization (secondary ablation)       mixed

Domain-specific parameters are ROWS OF SPARSE EMBEDDINGS looked up by the domain code and updated by
SparseAdam, which changes only the rows present in the batch (values and moments, i.e. lazy Adam). One dense
tensor with Adam would move domains absent from the batch through their old momentum (§16.4, review).

Unseen domains (no training row; e.g. UNKNOWN): every domain-specific term is replaced by its identity
(intercept 0, w_d = 0, b_d = 0, W_d = 1, gamma_d = 1, beta_d = 0, auxiliary logit 0) and PN uses the global
running statistics, so the output is the shared network's output. Row UNKNOWN of every categorical embedding
is zero (padding_idx), so an unseen value of any input contributes nothing.

PN statistics (implementation choices, §16.4): per-domain and global running mean / mean of squares, with
momentum max(0.1, 1/k) at the k-th update, so the first update replaces the initial value and a domain seen
in few batches is not pulled toward mean 0 / variance 1. A training batch with fewer than PN_MIN_ROWS rows is
normalized with the global running statistics and updates nothing; at evaluation a domain whose statistics
were never updated uses the global statistics (otherwise a near-zero variance from a handful of rows would
blow up new rows of that domain).

Run this module in a process that does NOT use LightGBM (see cvr/m3_worker.py for why).
"""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .mddata import MDArrays
from .schema import UNKNOWN

VARIANTS = ("int", "shared", "head", "star", "star_nopn")
SINGLE_DOMAIN_BATCHES = {"star"}
PN_MIN_ROWS = 256
PN_MOMENTUM = 0.1
PN_EPS = 1e-5


def _table(n: int, width: int, sparse: bool, fill: float | None) -> nn.Embedding:
    e = nn.Embedding(n, width, sparse=sparse)
    if fill is not None:
        with torch.no_grad():
            e.weight.fill_(fill)
    return e


class StarLinear(nn.Module):
    """out = x (W ⊙ W_d)ᵀ + (b + b_d) for one domain; the shared layer alone for an unseen domain."""

    def __init__(self, n_dom: int, d_in: int, d_out: int, sparse: bool):
        super().__init__()
        self.d_in, self.d_out = d_in, d_out
        self.shared = nn.Linear(d_in, d_out)
        self.dw = _table(n_dom, d_in * d_out, sparse, 1.0)
        self.db = _table(n_dom, d_out, sparse, 0.0)

    def forward(self, x: torch.Tensor, d: torch.Tensor, seen: bool) -> torch.Tensor:
        if not seen:
            return self.shared(x)
        w = self.shared.weight * self.dw(d).view(self.d_out, self.d_in)
        return x @ w.T + (self.shared.bias + self.db(d).view(self.d_out))


class PartitionedNorm(nn.Module):
    """z' = (gamma * gamma_d)(z - mu) / sqrt(var + eps) + (beta + beta_d)."""

    def __init__(self, n_dom: int, width: int, sparse: bool):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(width))
        self.beta = nn.Parameter(torch.zeros(width))
        self.dgamma = _table(n_dom, width, sparse, 1.0)
        self.dbeta = _table(n_dom, width, sparse, 0.0)
        self.register_buffer("dom_mean", torch.zeros(n_dom, width))
        self.register_buffer("dom_msq", torch.ones(n_dom, width))
        self.register_buffer("dom_updates", torch.zeros(n_dom, dtype=torch.long))
        self.register_buffer("glob_mean", torch.zeros(width))
        self.register_buffer("glob_msq", torch.ones(width))
        self.register_buffer("glob_updates", torch.zeros((), dtype=torch.long))

    @staticmethod
    def _var(mean: torch.Tensor, msq: torch.Tensor) -> torch.Tensor:
        return (msq - mean ** 2).clamp_min(0.0)

    @torch.no_grad()
    def _update(self, d: int, mu: torch.Tensor, msq: torch.Tensor) -> None:
        k = int(self.dom_updates[d]) + 1
        m = max(PN_MOMENTUM, 1.0 / k)
        self.dom_mean[d] = (1 - m) * self.dom_mean[d] + m * mu
        self.dom_msq[d] = (1 - m) * self.dom_msq[d] + m * msq
        self.dom_updates[d] = k
        k = int(self.glob_updates) + 1
        m = max(PN_MOMENTUM, 1.0 / k)
        self.glob_mean.mul_(1 - m).add_(m * mu)
        self.glob_msq.mul_(1 - m).add_(m * msq)
        self.glob_updates.fill_(k)

    def stats_source(self, d: int, seen: bool, n_rows: int) -> str:
        if self.training:
            return "batch" if (seen and n_rows >= PN_MIN_ROWS) else "global"
        return "domain" if (seen and int(self.dom_updates[d]) > 0) else "global"

    def forward(self, z: torch.Tensor, d: torch.Tensor, seen: bool) -> torch.Tensor:
        di = int(d)
        src = self.stats_source(di, seen, z.shape[0])
        if src == "batch":
            mu, var = z.mean(0), z.var(0, unbiased=False)
            self._update(di, mu.detach(), (z.detach() ** 2).mean(0))
        elif src == "domain":
            mu, var = self.dom_mean[di], self._var(self.dom_mean[di], self.dom_msq[di])
        else:
            mu, var = self.glob_mean, self._var(self.glob_mean, self.glob_msq)
        zn = (z - mu) / torch.sqrt(var + PN_EPS)
        if not seen:
            return zn * self.gamma + self.beta
        return zn * (self.gamma * self.dgamma(d).view(-1)) + (self.beta + self.dbeta(d).view(-1))


class MDNet(nn.Module):
    def __init__(self, cat_sizes: list[int], dom_col: int, n_num: int, variant: str, dim: int = 16,
                 hidden: tuple[int, ...] = (256, 128), aux_hidden: int = 32, sparse: bool = True):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant, self.dom_col = variant, dom_col
        n_dom = cat_sizes[dom_col]
        self.register_buffer("seen", torch.zeros(n_dom, dtype=torch.bool))  # domains with training rows
        self.emb = nn.ModuleList([nn.Embedding(s, dim, sparse=sparse, padding_idx=UNKNOWN) for s in cat_sizes])
        # int: the shared function never sees the domain; everything else gets the domain embedding as an input
        self.in_cols = [i for i in range(len(cat_sizes)) if not (variant == "int" and i == dom_col)]
        d_in = dim * len(self.in_cols) + n_num
        dims = [d_in, *hidden, 1]
        if variant in ("star", "star_nopn"):
            self.layers = nn.ModuleList([StarLinear(n_dom, a, b, sparse) for a, b in zip(dims[:-1], dims[1:])])
            self.pn = PartitionedNorm(n_dom, d_in, sparse) if variant == "star" else None
            self.aux_emb = _table(n_dom, dim, sparse, None)
            self.aux = nn.Sequential(nn.Linear(dim, aux_hidden), nn.ReLU(), nn.Linear(aux_hidden, 1))
        else:
            body = []
            for a, b in zip(dims[:-2], dims[1:-1]):
                body += [nn.Linear(a, b), nn.ReLU()]
            self.body = nn.Sequential(*body)
            self.out = nn.Linear(dims[-2], 1)
            if variant == "int":
                self.dom_bias = _table(n_dom, 1, sparse, 0.0)
            if variant == "head":
                self.dom_w = _table(n_dom, dims[-2], sparse, 0.0)
                self.dom_b = _table(n_dom, 1, sparse, 0.0)

    def inputs(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.emb[i](cat[:, i]) for i in self.in_cols] + [num], dim=-1)

    def _star_one(self, x: torch.Tensor, d: torch.Tensor, seen: bool) -> torch.Tensor:
        z = self.pn(x, d, seen) if self.pn is not None else x
        for j, layer in enumerate(self.layers):
            z = layer(z, d, seen)
            if j < len(self.layers) - 1:
                z = torch.relu(z)
        s = z.squeeze(-1)
        if seen:
            s = s + self.aux(self.aux_emb(d)).view(())
        return s

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        dom = cat[:, self.dom_col]
        x = self.inputs(cat, num)
        if self.variant in ("star", "star_nopn"):
            out = x.new_zeros(x.shape[0])
            for d in torch.unique(dom):  # one iteration for a single-domain training batch
                idx = (dom == d).nonzero().squeeze(1)
                dd = d.view(1)
                out = out.index_put((idx,), self._star_one(x[idx], dd, bool(self.seen[d])))
            return out
        h = self.body(x)
        out = self.out(h).squeeze(-1)
        seen = self.seen[dom].to(out.dtype)
        if self.variant == "int":
            out = out + self.dom_bias(dom).squeeze(-1) * seen
        elif self.variant == "head":
            out = out + ((self.dom_w(dom) * h).sum(-1) + self.dom_b(dom).squeeze(-1)) * seen
        return out


# ---------------------------------------------------------------- batching and optimisation
def mixed_batches(rng: np.random.Generator, n: int, batch: int) -> list[np.ndarray]:
    order = rng.permutation(n)
    return [order[s:s + batch] for s in range(0, n, batch)]


def domain_batches(rng: np.random.Generator, dom_rows: list[np.ndarray], batch: int) -> list[np.ndarray]:
    """One domain per batch (STAR). Each domain's rows are shuffled and cut into `batch`-sized chunks; a shorter
    last chunk is merged into the previous one, so no batch is smaller than min(batch, domain rows). The batch
    order is shuffled, so domains are visited in proportion to their size (implementation choice, §16.4)."""
    out = []
    for rows in dom_rows:
        rows = rows[rng.permutation(rows.size)]
        chunks = [rows[s:s + batch] for s in range(0, rows.size, batch)]
        if len(chunks) > 1 and chunks[-1].size < batch:
            chunks[-2] = np.concatenate([chunks[-2], chunks[-1]])
            chunks.pop()
        out += chunks
    return [out[i] for i in rng.permutation(len(out))]


def make_optimizers(net: nn.Module, lr: float) -> list[torch.optim.Optimizer]:
    """SparseAdam for sparse embeddings (lazy: rows absent from the batch keep values AND moments), Adam for the
    rest. With sparse=False every parameter goes to Adam (used by the test that shows the momentum leak)."""
    sparse_p = [m.weight for m in net.modules() if isinstance(m, nn.Embedding) and m.sparse]
    ids = {id(p) for p in sparse_p}
    dense_p = [p for p in net.parameters() if id(p) not in ids]
    opts: list[torch.optim.Optimizer] = [torch.optim.Adam(dense_p, lr=lr)]
    if sparse_p:
        opts.append(torch.optim.SparseAdam(sparse_p, lr=lr))
    return opts


def train_step(net: nn.Module, opts: list, cat: torch.Tensor, num: torch.Tensor, y: torch.Tensor,
               loss_fn: nn.Module) -> torch.Tensor:
    loss = loss_fn(net(cat, num), y)
    for o in opts:
        o.zero_grad(set_to_none=True)
    loss.backward()
    for o in opts:
        o.step()
    return loss.detach()


# ---------------------------------------------------------------- fit / predict
@dataclass
class MDModel:
    net: MDNet
    num_mean: np.ndarray
    num_std: np.ndarray
    info: dict = field(default_factory=dict)

    def tensors(self, a: MDArrays, idx: np.ndarray | slice) -> tuple[torch.Tensor, torch.Tensor]:
        cat = torch.as_tensor(np.asarray(a.cat[idx]), dtype=torch.long)
        num = torch.as_tensor((np.asarray(a.num[idx]) - self.num_mean) / self.num_std, dtype=torch.float32)
        return cat, num

    @classmethod
    def fit(cls, a: MDArrays, y: np.ndarray, cat_sizes: list[int], dom_col: int, variant: str, seed: int,
            dim: int = 16, epochs: int = 2, batch: int = 4096, lr: float = 1e-3, threads: int = 8,
            sparse: bool = True) -> list["MDModel"]:
        """Returns one model per epoch (checkpoint after epoch 1, 2, ...). The learning rate is constant and the
        random streams of epoch e do not depend on the total number of epochs, so the epoch-1 checkpoint of a
        2-epoch run equals a 1-epoch run (tests/test_mdnet.py)."""
        torch.manual_seed(seed)
        torch.set_num_threads(threads)
        rng = np.random.default_rng(seed)
        net = MDNet(cat_sizes, dom_col, a.num.shape[1], variant, dim=dim, sparse=sparse)
        dom = np.asarray(a.cat[:, dom_col])
        counts = np.bincount(dom, minlength=cat_sizes[dom_col])
        net.seen.copy_(torch.as_tensor(counts > 0))
        mean = a.num.mean(0).astype(np.float32)
        std = (a.num.std(0) + 1e-6).astype(np.float32)
        model = cls(net, mean, std)
        opts = make_optimizers(net, lr)
        loss_fn = nn.BCEWithLogitsLoss()
        yt = torch.as_tensor(y, dtype=torch.float32)
        dom_rows = [np.flatnonzero(dom == d) for d in np.flatnonzero(counts)]
        checkpoints, losses = [], []
        net.train()
        for ep in range(epochs):
            batches = (domain_batches(rng, dom_rows, batch) if variant in SINGLE_DOMAIN_BATCHES
                       else mixed_batches(rng, a.n, batch))
            for idx in batches:
                cat, num = model.tensors(a, idx)
                losses.append(train_step(net, opts, cat, num, yt[idx], loss_fn))
            info = {"variant": variant, "seed": seed, "dim": dim, "epochs": ep + 1, "batch": batch, "lr": lr,
                    "steps": len(losses), "train_loss_last100_mean": float(torch.stack(losses[-100:]).mean()),
                    "params": int(sum(p.numel() for p in net.parameters())),
                    "domains_seen": int((counts > 0).sum())}
            checkpoints.append(cls(copy.deepcopy(net).eval(), mean, std, info))
        return checkpoints

    @torch.no_grad()
    def predict(self, a: MDArrays, batch: int = 16384) -> np.ndarray:
        self.net.eval()
        out = np.empty(a.n, dtype=np.float64)
        for s in range(0, a.n, batch):
            idx = slice(s, min(s + batch, a.n))
            out[idx] = torch.sigmoid(self.net(*self.tensors(a, idx))).numpy()
        return out


def save_md_model(m: MDModel, cat_sizes: list[int], dom_col: int, path: Path) -> None:
    torch.save(m.net.state_dict(), path.with_suffix(".pt"))
    np.savez(path.with_suffix(".npz"), num_mean=m.num_mean, num_std=m.num_std)
    path.with_suffix(".json").write_text(json.dumps({"cat_sizes": cat_sizes, "dom_col": dom_col, "info": m.info}))


def load_md_model(path: Path) -> MDModel:
    meta = json.loads(path.with_suffix(".json").read_text())
    z = np.load(path.with_suffix(".npz"))
    info = meta["info"]
    net = MDNet(meta["cat_sizes"], meta["dom_col"], n_num=len(z["num_mean"]), variant=info["variant"],
                dim=info["dim"], sparse=False)
    net.load_state_dict(torch.load(path.with_suffix(".pt"), map_location="cpu"))
    return MDModel(net.eval(), z["num_mean"], z["num_std"], info)
