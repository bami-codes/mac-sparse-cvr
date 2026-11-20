"""Study C networks (docs/DESIGN.md §16.4): the M2 inputs, five ways of using the domain.

Domain = the DOMAIN_KEY (context_feat1) code. Inputs: CAT_INPUTS codes (embedded) + standardized NUM_INPUTS.

variant     how the domain is used                                                    optimizer steps
int         no domain input; a per-domain logit intercept                             mixed batches
shared      the domain embedding is one input of a shared MLP                         mixed batches
head        shared body (domain is an input) + output (w + w_d)·h + (b + b_d)          mixed batches
star        STAR (Sheng et al., CIKM 2021): partitioned normalization (PN) on the     single-domain chunks,
            concatenated input, star-topology FCN in every layer (W ⊙ W_d, b + b_d),  accumulated to >= batch
            auxiliary 2-layer network on the domain embedding added to the logit       rows per step
star_nopn   star FCN + auxiliary network, no normalization (secondary ablation)       mixed batches

Domain-specific parameters are rows of sparse embeddings looked up by the domain code and updated by
SparseAdam, which changes only the rows present in the batch (values and moments, i.e. lazy Adam). One dense
tensor with Adam would keep moving domains absent from the batch through their old momentum (§16.4).

Unseen domains (no training row; UNKNOWN, or a code that never occurs in training) are mapped to UNKNOWN before
anything else, including the shared input embedding (whose UNKNOWN row is zero, padding_idx). Every
domain-specific term then takes its identity (intercept 0, w_d = 0, b_d = 0, W_d = 1, gamma_d = 1, beta_d = 0,
auxiliary logit 0) and PN uses the global running statistics, so every unseen domain gets the same shared
prediction (§16.4 revision 2: before, the shared input still carried a random embedding of the original code).

Numeric inputs: standardized with training mean / std; a column constant over the training rows is set to the
neutral 0 at training AND inference (a new value there would otherwise be divided by ~1e-6), and the std is
floored at NUM_STD_FLOOR.

PN statistics (implementation choices, §16.4):
- per-domain and global running mean / mean of squares, momentum max(0.1, 1/k) at the k-th update, so the first
  update replaces the initial value;
- one training forward = one optimizer step, which may hold several domains (accumulated chunks). The global
  statistics are read once at the start of the step, so every domain of the step refers to the same values, and
  updated once at the end from the sum, sum of squares and count of all the step's rows (revision 3: updating
  them while looping over the domains in code order made the result depend on the codes, which follow traffic
  rank, and averaged a 4,096-row and a 512-row domain with equal weight);
- a training batch with fewer than PN_MIN_ROWS rows of a domain is normalized with the global statistics and
  updates nothing; at evaluation a domain whose statistics were never updated uses the global statistics;
- PER DIMENSION, a domain (or batch) variance below PN_REL_VAR x the global variance is replaced by the global
  mean and variance of that dimension: enough rows do not make every feature vary (e.g. a purchase-history
  match that is always 0 in one group), and dividing by that near-zero variance blew up a later 1;
- the global variance is floored at PN_VAR_FLOOR;
- the dimensions of the DOMAIN-ID embedding always use the global statistics, in training and evaluation
  (revision 4). Within a domain that embedding is constant at any one step, so its own variance is 0 while
  training, but the running mean / mean of squares mix values from different steps, so at evaluation its variance
  could pass the 1% rule and switch the normalization (one counterexample gave -0.577 in training and +1.000 in
  evaluation for the same state and input).

Run this module in a process that does not use LightGBM (see cvr/m3_worker.py for why).
"""

import copy
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .mddata import MDArrays
from .schema import UNKNOWN

VARIANTS = ("int", "shared", "head", "star", "star_nopn")
SINGLE_DOMAIN_CHUNKS = {"star"}
PN_MIN_ROWS = 256
PN_MOMENTUM = 0.1
PN_EPS = 1e-5
PN_REL_VAR = 0.01  # own variance must be >= 1% of the global variance (own std >= 10% of the global std)
PN_VAR_FLOOR = 1e-2
NUM_STD_FLOOR = 1e-2


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
        # dimensions that always use the global statistics (set by MDNet: the domain-ID embedding); not saved
        self.register_buffer("global_dims", torch.zeros(width, dtype=torch.bool), persistent=False)
        self._snapshot: tuple[torch.Tensor, torch.Tensor] | None = None  # global stats read at the step start
        self._acc: list | None = None  # [rows, sum, sum of squares] of the step, for the end-of-step update

    @staticmethod
    def _var(mean: torch.Tensor, msq: torch.Tensor) -> torch.Tensor:
        return (msq - mean**2).clamp_min(0.0)

    def global_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.glob_mean.clone(), self._var(self.glob_mean, self.glob_msq).clamp_min(PN_VAR_FLOOR)

    def begin_step(self) -> None:
        self._snapshot = self.global_stats()
        self._acc = [0, None, None]

    @torch.no_grad()
    def end_step(self) -> None:
        n, s, ss = self._acc if self._acc is not None else (0, None, None)
        if n >= PN_MIN_ROWS:
            k = int(self.glob_updates) + 1
            m = max(PN_MOMENTUM, 1.0 / k)
            self.glob_mean.mul_(1 - m).add_(m * (s / n).to(self.glob_mean.dtype))
            self.glob_msq.mul_(1 - m).add_(m * (ss / n).to(self.glob_msq.dtype))
            self.glob_updates.fill_(k)
        self._snapshot, self._acc = None, None

    @torch.no_grad()
    def _accumulate(self, z: torch.Tensor) -> None:
        if self._acc is None:
            return
        zd = z.detach().double()
        n, s, ss = self._acc
        self._acc = [
            n + z.shape[0],
            zd.sum(0) if s is None else s + zd.sum(0),
            (zd**2).sum(0) if ss is None else ss + (zd**2).sum(0),
        ]

    @torch.no_grad()
    def _update_domain(self, d: int, mu: torch.Tensor, msq: torch.Tensor) -> None:
        k = int(self.dom_updates[d]) + 1
        m = max(PN_MOMENTUM, 1.0 / k)
        self.dom_mean[d] = (1 - m) * self.dom_mean[d] + m * mu
        self.dom_msq[d] = (1 - m) * self.dom_msq[d] + m * msq
        self.dom_updates[d] = k

    def stats_source(self, d: int, seen: bool, n_rows: int) -> str:
        if self.training:
            return "batch" if (seen and n_rows >= PN_MIN_ROWS) else "global"
        return "domain" if (seen and int(self.dom_updates[d]) > 0) else "global"

    def stats(self, z: torch.Tensor, d: int, seen: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """(mu, var) per dimension. In a training step the global reference is the step-start snapshot, the
        domain's own statistics are updated from its rows, and the rows are added to the end-of-step global sums."""
        use_snapshot = self.training and self._snapshot is not None
        g_mu, g_var = self._snapshot if use_snapshot else self.global_stats()
        if self.training:
            self._accumulate(z)
        src = self.stats_source(d, seen, z.shape[0])
        if src == "global":
            return g_mu, g_var
        if src == "batch":
            mu, var = z.mean(0), z.var(0, unbiased=False)
            self._update_domain(d, mu.detach(), (z.detach() ** 2).mean(0))
        else:
            mu, var = self.dom_mean[d].clone(), self._var(self.dom_mean[d], self.dom_msq[d])
        # a (near-)constant dimension of this domain, or a domain-ID dimension, uses the global statistics
        own = (var.detach() >= PN_REL_VAR * g_var) & ~self.global_dims
        return torch.where(own, mu, g_mu), torch.where(own, var, g_var)

    def forward(self, z: torch.Tensor, d: torch.Tensor, seen: bool) -> torch.Tensor:
        mu, var = self.stats(z, int(d), seen)
        zn = (z - mu) / torch.sqrt(var + PN_EPS)
        if not seen:
            return zn * self.gamma + self.beta
        return zn * (self.gamma * self.dgamma(d).view(-1)) + (self.beta + self.dbeta(d).view(-1))


class MDNet(nn.Module):
    def __init__(
        self,
        cat_sizes: list[int],
        dom_col: int,
        n_num: int,
        variant: str,
        dim: int = 16,
        hidden: tuple[int, ...] = (256, 128),
        aux_hidden: int = 32,
        sparse: bool = True,
    ):
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
            self.layers = nn.ModuleList([StarLinear(n_dom, a, b, sparse) for a, b in itertools.pairwise(dims)])
            self.pn = PartitionedNorm(n_dom, d_in, sparse) if variant == "star" else None
            if self.pn is not None:
                start = self.in_cols.index(dom_col) * dim
                self.pn.global_dims[start : start + dim] = True
            self.aux_emb = _table(n_dom, dim, sparse, None)
            self.aux = nn.Sequential(nn.Linear(dim, aux_hidden), nn.ReLU(), nn.Linear(aux_hidden, 1))
        else:
            body = []
            for a, b in itertools.pairwise(dims[:-1]):  # hidden layers; the output layer follows
                body += [nn.Linear(a, b), nn.ReLU()]
            self.body = nn.Sequential(*body)
            self.out = nn.Linear(dims[-2], 1)
            if variant == "int":
                self.dom_bias = _table(n_dom, 1, sparse, 0.0)
            if variant == "head":
                self.dom_w = _table(n_dom, dims[-2], sparse, 0.0)
                self.dom_b = _table(n_dom, 1, sparse, 0.0)

    def domain(self, cat: torch.Tensor) -> torch.Tensor:
        """The domain code used everywhere: codes without training rows become UNKNOWN (seen[UNKNOWN] is False)."""
        dom = cat[:, self.dom_col]
        return torch.where(self.seen[dom], dom, torch.full_like(dom, UNKNOWN))

    def inputs(self, cat: torch.Tensor, num: torch.Tensor, dom: torch.Tensor) -> torch.Tensor:
        cols = [dom if i == self.dom_col else cat[:, i] for i in self.in_cols]
        return torch.cat([self.emb[i](c) for i, c in zip(self.in_cols, cols)] + [num], dim=-1)

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
        dom = self.domain(cat)
        x = self.inputs(cat, num, dom)
        if self.variant in ("star", "star_nopn"):
            step = self.training and self.pn is not None
            if step:
                self.pn.begin_step()
            try:
                out = x.new_zeros(x.shape[0])
                for d in torch.unique(dom):  # PN statistics and star weights are per domain
                    idx = (dom == d).nonzero().squeeze(1)
                    out = out.index_put((idx,), self._star_one(x[idx], d.view(1), bool(self.seen[d])))
            finally:
                if step:
                    self.pn.end_step()
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
    return [order[s : s + batch] for s in range(0, n, batch)]


def domain_batches(rng: np.random.Generator, dom_rows: list[np.ndarray], batch: int) -> list[np.ndarray]:
    """Single-domain chunks (STAR). Each domain's rows are shuffled and cut into `batch`-sized chunks; a shorter
    last chunk is merged into the previous one, so no chunk is smaller than min(batch, domain rows). The chunk
    order is shuffled."""
    out = []
    for rows in dom_rows:
        rows = rows[rng.permutation(rows.size)]
        chunks = [rows[s : s + batch] for s in range(0, rows.size, batch)]
        if len(chunks) > 1 and chunks[-1].size < batch:
            chunks[-2] = np.concatenate([chunks[-2], chunks[-1]])
            chunks.pop()
        out += chunks
    return [out[i] for i in rng.permutation(len(out))]


def accumulate_steps(chunks: list[np.ndarray], batch: int) -> list[np.ndarray]:
    """Optimizer steps of >= `batch` rows built from consecutive single-domain chunks (a trailing short step is
    merged into the previous one). The loss of a step is the mean over its rows, so every click has about the
    same weight (step sizes stay within [batch, 4·batch)) and a 5-row domain does not get a full update of its
    own, as it would with one step per chunk. A domain appears in at
    most one chunk of a step, so its PN statistics are still computed on that domain's rows only."""
    steps, cur, n = [], [], 0
    for c in chunks:
        cur.append(c)
        n += c.size
        if n >= batch:
            steps.append(np.concatenate(cur))
            cur, n = [], 0
    if cur:
        tail = np.concatenate(cur)
        if steps:
            steps[-1] = np.concatenate([steps[-1], tail])
        else:
            steps.append(tail)
    return steps


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


def train_step(
    net: nn.Module, opts: list, cat: torch.Tensor, num: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module
) -> torch.Tensor:
    loss = loss_fn(net(cat, num), y)  # mean over the step's rows
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
    num_std: np.ndarray  # floored at NUM_STD_FLOOR
    num_keep: np.ndarray  # 0 for columns constant over the training rows (always fed as the neutral 0)
    info: dict = field(default_factory=dict)

    def tensors(self, a: MDArrays, idx: np.ndarray | slice) -> tuple[torch.Tensor, torch.Tensor]:
        cat = torch.as_tensor(np.array(a.cat[idx], dtype=np.int64))  # a copy: memory-mapped slices are read-only
        z = (np.asarray(a.num[idx]) - self.num_mean) / self.num_std * self.num_keep
        return cat, torch.as_tensor(z, dtype=torch.float32)

    @staticmethod
    def numeric_scaler(num: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(num, dtype=np.float64)
        mean, std = x.mean(0), x.std(0)
        keep = (std > 0).astype(np.float32)
        return mean.astype(np.float32), np.maximum(std, NUM_STD_FLOOR).astype(np.float32), keep

    @classmethod
    def fit(
        cls,
        a: MDArrays,
        y: np.ndarray,
        cat_sizes: list[int],
        dom_col: int,
        variant: str,
        seed: int,
        dim: int = 16,
        epochs: int = 2,
        batch: int = 4096,
        lr: float = 1e-3,
        threads: int = 8,
        sparse: bool = True,
    ) -> list["MDModel"]:
        """Returns one model per epoch (checkpoint after epoch 1, 2, ...). The learning rate is constant and the
        random streams of epoch e do not depend on the total number of epochs, so the epoch-1 checkpoint of a
        2-epoch run equals a 1-epoch run (tests/test_mdnet.py)."""
        torch.manual_seed(seed)
        torch.set_num_threads(threads)
        rng = np.random.default_rng(seed)
        net = MDNet(cat_sizes, dom_col, a.num.shape[1], variant, dim=dim, sparse=sparse)
        dom = np.asarray(a.cat[:, dom_col])
        counts = np.bincount(dom, minlength=cat_sizes[dom_col])
        if counts[UNKNOWN]:
            raise ValueError("training rows with an UNKNOWN domain code: the vocabulary must come from these rows")
        net.seen.copy_(torch.as_tensor(counts > 0))
        model = cls(net, *cls.numeric_scaler(a.num))
        opts = make_optimizers(net, lr)
        loss_fn = nn.BCEWithLogitsLoss()
        yt = torch.as_tensor(y, dtype=torch.float32)
        dom_ids = np.flatnonzero(counts)
        dom_rows = [np.flatnonzero(dom == d) for d in dom_ids]
        checkpoints, losses, step_rows, dom_steps = [], [], [], np.zeros(len(counts), dtype=np.int64)
        net.train()
        for ep in range(epochs):
            if variant in SINGLE_DOMAIN_CHUNKS:
                steps = accumulate_steps(domain_batches(rng, dom_rows, batch), batch)
            else:
                steps = mixed_batches(rng, a.n, batch)
            for idx in steps:
                cat, num = model.tensors(a, idx)
                losses.append(train_step(net, opts, cat, num, yt[idx], loss_fn))
                if ep == 0:
                    step_rows.append(idx.size)
                    dom_steps[np.unique(dom[idx])] += 1
            info = {
                "variant": variant,
                "seed": seed,
                "dim": dim,
                "epochs": ep + 1,
                "batch": batch,
                "lr": lr,
                "steps": len(losses),
                "train_loss_last100_mean": float(torch.stack(losses[-100:]).mean()),
                "params": int(sum(p.numel() for p in net.parameters())),
                "domains_seen": int((counts > 0).sum()),
                "num_constant_columns": int((model.num_keep == 0).sum()),
                "pn_global_only_dims": int(net.pn.global_dims.sum()) if getattr(net, "pn", None) is not None else None,
                "step_rows_epoch1": {
                    "min": int(min(step_rows)),
                    "median": float(np.median(step_rows)),
                    "max": int(max(step_rows)),
                },
                "domain_rows": {int(d): int(counts[d]) for d in dom_ids},
                "domain_steps_epoch1": {int(d): int(dom_steps[d]) for d in dom_ids},
            }
            checkpoints.append(cls(copy.deepcopy(net).eval(), model.num_mean, model.num_std, model.num_keep, info))
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
    np.savez(path.with_suffix(".npz"), num_mean=m.num_mean, num_std=m.num_std, num_keep=m.num_keep)
    path.with_suffix(".json").write_text(json.dumps({"cat_sizes": cat_sizes, "dom_col": dom_col, "info": m.info}))


def load_md_model(path: Path) -> MDModel:
    meta = json.loads(path.with_suffix(".json").read_text())
    z = np.load(path.with_suffix(".npz"))
    info = meta["info"]
    net = MDNet(
        meta["cat_sizes"],
        meta["dom_col"],
        n_num=len(z["num_mean"]),
        variant=info["variant"],
        dim=info["dim"],
        sparse=False,
    )
    net.load_state_dict(torch.load(path.with_suffix(".pt"), map_location="cpu"))
    return MDModel(net.eval(), z["num_mean"], z["num_std"], z["num_keep"], info)
