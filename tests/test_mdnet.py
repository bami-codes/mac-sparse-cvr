"""Synthetic checks for the study C networks (docs/DESIGN.md §16.8, tests 1-4, 6, 7). No MAC data needed."""

import numpy as np
import pytest
import torch
from torch import nn

from cvr.mddata import MDArrays
from cvr.mdnet import (PN_EPS, PN_MIN_ROWS, MDModel, MDNet, PartitionedNorm, domain_batches, load_md_model,
                       make_optimizers, save_md_model, train_step)
from cvr.schema import UNKNOWN

CAT_SIZES = [7, 10, 20]  # column 0 is the domain: codes 2..5 have training rows; 0 (UNKNOWN), 1 and 6 have none
DOM_COL = 0
ROWS = {2: 1500, 3: 600, 4: 300, 5: 3}  # domain 5 is below PN_MIN_ROWS
UNSEEN = [UNKNOWN, 6]


def _data(seed=0):
    rng = np.random.default_rng(seed)
    dom = np.concatenate([np.full(n, d) for d, n in ROWS.items()])
    rng.shuffle(dom)
    n = dom.size
    cat = np.column_stack([dom, rng.integers(2, 10, n), rng.integers(2, 20, n)]).astype(np.int32)
    num = rng.normal(size=(n, 4)).astype(np.float32)
    logit = -2.5 + 0.8 * num[:, 0] + 0.6 * (dom == 3) * num[:, 1] - 0.5 * (cat[:, 1] == 4)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.float32)
    return MDArrays(cat, num), y


def _fit(variant, epochs=1, seed=1, **kw):
    a, y = _data()
    return MDModel.fit(a, y, CAT_SIZES, DOM_COL, variant, seed=seed, dim=4, epochs=epochs, batch=256, lr=3e-3,
                       threads=1, **kw)


def _rows_with_domain(d, k=50, seed=3):
    a, _ = _data(seed)
    sub = a.take(np.arange(k))
    sub.cat[:, DOM_COL] = d
    return sub


# ---------------------------------------------------------------- 1. partitioned normalization
def test_single_row_training_batch_is_finite_and_updates_nothing():
    torch.manual_seed(0)
    net = MDNet(CAT_SIZES, DOM_COL, 4, "star", dim=4)
    net.seen[2:6] = True
    net.train()
    a, _ = _data()
    cat = torch.as_tensor(a.cat[:1], dtype=torch.long)
    out = net(cat, torch.as_tensor(a.num[:1]))
    assert torch.isfinite(out).all()
    assert int(net.pn.dom_updates.sum()) == 0 and int(net.pn.glob_updates) == 0  # below PN_MIN_ROWS: global stats


def test_domain_batches_are_single_domain_cover_every_row_once_and_are_never_tiny():
    a, _ = _data()
    dom = a.cat[:, DOM_COL]
    dom_rows = [np.flatnonzero(dom == d) for d in sorted(ROWS)]
    batches = domain_batches(np.random.default_rng(0), dom_rows, 256)
    assert np.array_equal(np.sort(np.concatenate(batches)), np.arange(a.n))
    for b in batches:
        d = np.unique(dom[b])
        assert d.size == 1
        assert b.size >= min(256, ROWS[int(d[0])])


def test_first_pn_update_replaces_the_initial_statistics():
    pn = PartitionedNorm(7, 3, sparse=True).train()
    z = torch.randn(300, 3) * 2 + 5
    pn(z, torch.tensor([4]), True)
    assert torch.allclose(pn.dom_mean[4], z.mean(0), atol=1e-5)
    assert torch.allclose(pn._var(pn.dom_mean[4], pn.dom_msq[4]), z.var(0, unbiased=False), atol=1e-4)
    assert int(pn.dom_updates[4]) == 1


def test_domain_never_updated_uses_global_statistics_at_evaluation():
    m = _fit("star")[0]
    pn = m.net.pn
    assert int(pn.dom_updates[5]) == 0  # 3 training rows < PN_MIN_ROWS
    assert pn.stats_source(5, True, 10) == "global"
    assert pn.stats_source(2, True, 10) == "domain"
    assert np.isfinite(m.predict(_rows_with_domain(5))).all()


# ---------------------------------------------------------------- 2. unseen domains after training
@pytest.mark.parametrize("variant", ["int", "head", "star", "star_nopn"])
def test_unseen_domain_follows_the_shared_path_after_training(variant):
    m = _fit(variant)[0]
    net = m.net
    for d in UNSEEN:
        sub = _rows_with_domain(d)
        got = m.predict(sub)
        cat, num = m.tensors(sub, slice(None))
        with torch.no_grad():
            x = net.inputs(cat, num)
            if variant in ("int", "head"):
                expected = net.out(net.body(x)).squeeze(-1)
            else:
                z = x
                if net.pn is not None:
                    var = net.pn._var(net.pn.glob_mean, net.pn.glob_msq)
                    z = (z - net.pn.glob_mean) / torch.sqrt(var + PN_EPS) * net.pn.gamma + net.pn.beta
                for j, layer in enumerate(net.layers):
                    z = layer.shared(z)
                    if j < len(net.layers) - 1:
                        z = torch.relu(z)
                expected = z.squeeze(-1)
        assert np.array_equal(got, torch.sigmoid(expected).numpy().astype(np.float64)), d
        # the unseen domain's own parameter rows are still at their initial (identity) values
        with torch.no_grad():
            if variant == "int":
                assert float(net.dom_bias.weight[d].abs().max()) == 0.0
            if variant == "head":
                assert float(net.dom_w.weight[d].abs().max()) == 0.0 and float(net.dom_b.weight[d].abs().max()) == 0.0
            if variant in ("star", "star_nopn"):
                for layer in net.layers:
                    assert bool((layer.dw.weight[d] == 1).all()) and bool((layer.db.weight[d] == 0).all())
            if variant == "star":
                assert bool((net.pn.dgamma.weight[d] == 1).all()) and bool((net.pn.dbeta.weight[d] == 0).all())


# ---------------------------------------------------------------- 3. optimizer isolation of domain parameters
def _domain_rows(net, d):
    rows = {}
    for name, mod in net.named_modules():
        if isinstance(mod, nn.Embedding) and mod.weight.shape[0] == CAT_SIZES[DOM_COL] and not name.startswith("emb."):
            rows[name] = mod.weight.detach()[d].clone()
    if getattr(net, "pn", None) is not None:
        rows["pn.dom_mean"] = net.pn.dom_mean[d].clone()
    return rows


def _isolation(variant, sparse):
    torch.manual_seed(0)
    net = MDNet(CAT_SIZES, DOM_COL, 4, variant, dim=4, sparse=sparse)
    net.seen[2:6] = True
    net.train()
    opts = make_optimizers(net, 1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    a, y = _data()

    def batch(d):
        idx = np.flatnonzero(a.cat[:, DOM_COL] == d)[:300]
        return (torch.as_tensor(a.cat[idx], dtype=torch.long), torch.as_tensor(a.num[idx]),
                torch.as_tensor(y[idx]))

    A, B = 2, 3
    train_step(net, opts, *batch(A), loss_fn)
    before = _domain_rows(net, A)
    cat, num, yb = batch(B)
    loss = loss_fn(net(cat, num), yb)
    for o in opts:
        o.zero_grad(set_to_none=True)
    loss.backward()
    grad_rows_zero = True
    for name, mod in net.named_modules():
        if name in before and mod.weight.grad is not None:
            g = mod.weight.grad
            if g.is_sparse:
                grad_rows_zero &= A not in g.coalesce().indices()[0].tolist()
            else:
                grad_rows_zero &= bool((g[A] == 0).all())
    for o in opts:
        o.step()
    after = _domain_rows(net, A)
    moved = any(not torch.equal(before[k], after[k]) for k in before)
    return grad_rows_zero, moved, list(before)


@pytest.mark.parametrize("variant", ["star", "head", "int"])
def test_a_batch_of_domain_b_does_not_move_domain_a(variant):
    grad_zero, moved, names = _isolation(variant, sparse=True)
    assert names, "no domain parameters found"
    assert grad_zero and not moved


@pytest.mark.parametrize("variant", ["star", "head"])
def test_dense_adam_would_move_domain_a_through_momentum(variant):
    """The defect the sparse optimizer avoids: the gradient row is zero, yet Adam's momentum moves the row."""
    grad_zero, moved, _ = _isolation(variant, sparse=False)
    assert grad_zero and moved


# ---------------------------------------------------------------- 4. int never sees the domain
def test_int_variant_uses_the_domain_only_through_its_intercept():
    m = _fit("int")[0]
    assert DOM_COL not in m.net.in_cols
    s2, s3 = _rows_with_domain(2), _rows_with_domain(3)
    with torch.no_grad():
        l2 = m.net(*m.tensors(s2, slice(None)))
        l3 = m.net(*m.tensors(s3, slice(None)))
        gap = float(m.net.dom_bias.weight[2] - m.net.dom_bias.weight[3])
    assert torch.allclose(l2 - l3, torch.full_like(l2, gap), atol=1e-6)


# ---------------------------------------------------------------- 6. determinism, reload, mixed evaluation batches
@pytest.mark.parametrize("variant", ["star", "head", "int"])
def test_seed_determinism_and_exact_reload(tmp_path, variant):
    a, _ = _data()
    p1 = _fit(variant)[0].predict(a)
    m = _fit(variant)[0]
    p2 = m.predict(a)
    assert np.array_equal(p1, p2)
    assert np.isfinite(p1).all() and ((p1 > 0) & (p1 < 1)).all()
    save_md_model(m, CAT_SIZES, DOM_COL, tmp_path / "m")
    assert np.array_equal(load_md_model(tmp_path / "m").predict(a), p2)


def test_star_mixed_evaluation_batch_equals_per_domain_evaluation():
    m = _fit("star")[0]
    a, _ = _data()
    full = m.predict(a)
    for d in ROWS:
        idx = np.flatnonzero(a.cat[:, DOM_COL] == d)
        assert np.array_equal(m.predict(a.take(idx)), full[idx])


# ---------------------------------------------------------------- 7. epoch checkpoints
@pytest.mark.parametrize("variant", ["star", "shared"])
def test_epoch1_checkpoint_of_a_two_epoch_run_equals_a_one_epoch_run(variant):
    a, _ = _data()
    two = _fit(variant, epochs=2)
    one = _fit(variant, epochs=1)
    assert len(two) == 2 and two[0].info["epochs"] == 1 and two[1].info["epochs"] == 2
    assert np.array_equal(two[0].predict(a), one[0].predict(a))
    assert not np.array_equal(two[0].predict(a), two[1].predict(a))


def test_pn_min_rows_is_below_the_small_group_threshold():
    # every small group of the primary metric (>= 3,000 training clicks, §16.3) gets its own PN statistics
    assert PN_MIN_ROWS <= 3000
