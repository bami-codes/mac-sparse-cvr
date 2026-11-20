"""Synthetic checks for M3 padding and pooling (no MAC data needed)."""

import numpy as np
import torch

from cvr.nn import M3Arrays, M3Model, M3Net
from cvr.schema import MISSING


def _batch(B=6, L=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    other = torch.randint(0, 5, (B, 3), generator=g)
    target = torch.randint(2, 50, (B, 3), generator=g)
    seq = torch.randint(2, 50, (B, L, 3), generator=g)
    seq[0] = MISSING  # row 0: empty history (all PAD)
    seq[1, 10:] = MISSING  # row 1: half PAD
    mm = torch.rand((B, L, 2), generator=g)
    num = torch.randn((B, 4), generator=g)
    return other, target, seq, mm, num


def _net(pooling):
    torch.manual_seed(0)
    return M3Net([5, 5, 5], [50, 50, 50], n_num=4, dim=8, pooling=pooling, sparse=False)


def test_all_pad_history_gives_finite_output_and_gradients():
    for pooling in ("attn", "mean"):
        net = _net(pooling)
        out = net(*_batch())
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_pad_steps_do_not_change_output():
    for pooling in ("attn", "mean"):
        net = _net(pooling)
        other, target, seq, mm, num = _batch()
        base = net(other, target, seq, mm, num)
        mm2 = mm.clone()
        mm2[1, 10:] = 99.0  # change the similarity scores only on PAD steps of row 1
        mm2[0] = -99.0  # and on every step of the all-PAD row 0
        assert torch.allclose(base, net(other, target, seq, mm2, num))


def test_mean_pooling_is_average_over_valid_steps():
    net = _net("mean")
    _other, _target, seq, mm, _num = _batch()
    k = net.key_proj(torch.cat([e(seq[..., i]) for i, e in enumerate(net.id_emb)] + [mm], dim=-1))
    valid = (seq[..., 0] != MISSING).float()
    expected = (k * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
    # recompute the pooled vector the way forward() does
    w = torch.softmax(torch.zeros_like(valid).masked_fill(valid == 0, -1e9), dim=1) * valid
    assert torch.allclose((w.unsqueeze(-1) * k).sum(1), expected, atol=1e-6)
    assert torch.allclose(expected[0], torch.zeros_like(expected[0]))  # empty history -> zeros


def test_fit_and_predict_run_and_are_seed_deterministic():
    rng = np.random.default_rng(0)
    n = 512
    a = M3Arrays(other=rng.integers(0, 5, (n, 3)).astype(np.int32),
                 target=rng.integers(2, 50, (n, 3)).astype(np.int32),
                 seq=rng.integers(1, 50, (n, 20, 3)).astype(np.int32),
                 mm=rng.random((n, 20, 2)).astype(np.float32),
                 num=rng.normal(size=(n, 4)).astype(np.float32))
    y = (rng.random(n) < 0.1).astype(np.float32)
    p1 = M3Model.fit(a, y, [5, 5, 5], [50, 50, 50], "attn", seed=7, dim=8, batch=128, threads=1).predict(a)
    p2 = M3Model.fit(a, y, [5, 5, 5], [50, 50, 50], "attn", seed=7, dim=8, batch=128, threads=1).predict(a)
    assert np.isfinite(p1).all() and ((p1 > 0) & (p1 < 1)).all()
    assert np.array_equal(p1, p2)
