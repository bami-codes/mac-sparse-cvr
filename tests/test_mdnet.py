"""Synthetic checks for the study C networks (docs/DESIGN.md §16.8, tests 1-4, 6, 7). No MAC data needed."""

import numpy as np
import pytest
import torch
from torch import nn

from cvr.mddata import MDArrays
from cvr.checks import NN_SINGLE_VS_BATCH_TOL
from cvr.mdnet import (PN_EPS, PN_MIN_ROWS, MDModel, MDNet, PartitionedNorm, accumulate_steps, domain_batches,
                       load_md_model, make_optimizers, save_md_model, train_step)
from cvr.schema import MISSING, UNKNOWN

CAT_SIZES = [7, 10, 20]  # column 0 is the domain: codes 2..5 have training rows; 0 (UNKNOWN), 1 and 6 have none
DOM_COL = 0
ROWS = {2: 1500, 3: 600, 4: 300, 5: 3}  # domain 5 is below PN_MIN_ROWS
UNSEEN = [UNKNOWN, MISSING, 6]  # none has a training row


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
def _shared_path(net, cat, num):
    """The shared network written out by hand: the domain input is the zero (UNKNOWN) vector, no domain term."""
    parts = [torch.zeros(cat.shape[0], net.emb[i].embedding_dim) if i == DOM_COL else net.emb[i](cat[:, i])
             for i in net.in_cols]
    x = torch.cat(parts + [num], dim=-1)
    if net.variant in ("int", "shared", "head"):
        return net.out(net.body(x)).squeeze(-1)
    z = x
    if net.pn is not None:
        var = net.pn._var(net.pn.glob_mean, net.pn.glob_msq).clamp_min(1e-2)
        z = (z - net.pn.glob_mean) / torch.sqrt(var + PN_EPS) * net.pn.gamma + net.pn.beta
    for j, layer in enumerate(net.layers):
        z = layer.shared(z)
        if j < len(net.layers) - 1:
            z = torch.relu(z)
    return z.squeeze(-1)


@pytest.mark.parametrize("variant", ["int", "shared", "head", "star", "star_nopn"])
def test_unseen_domains_all_get_the_shared_prediction_after_training(variant):
    m = _fit(variant)[0]
    net = m.net
    preds = {}
    for d in UNSEEN:
        sub = _rows_with_domain(d)
        preds[d] = m.predict(sub)
        cat, num = m.tensors(sub, slice(None))
        with torch.no_grad():
            expected = torch.sigmoid(_shared_path(net, cat, num)).numpy().astype(np.float64)
        assert np.allclose(preds[d], expected, rtol=0, atol=1e-7), d
    # review counterexample: UNKNOWN and MISSING rows differed by up to 0.12 through the shared input embedding
    assert np.array_equal(preds[UNKNOWN], preds[MISSING]) and np.array_equal(preds[UNKNOWN], preds[6])
    with torch.no_grad():  # the unseen codes' own parameter rows are still at their initial (identity) values
        for d in UNSEEN:
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
    # Exact on macOS arm64; on a Linux x86 GPU pod the batch-size-dependent float32 matmul kernels differ by up to
    # 7.5e-9 (measured). Statistics leaking across rows in evaluation would differ by ~1e-2, far above 1e-6.
    from cvr.checks import NN_SINGLE_VS_BATCH_TOL
    m = _fit("star")[0]
    a, _ = _data()
    full = m.predict(a)
    for d in ROWS:
        idx = np.flatnonzero(a.cat[:, DOM_COL] == d)
        assert np.abs(m.predict(a.take(idx)) - full[idx]).max() <= NN_SINGLE_VS_BATCH_TOL


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


# ---------------------------------------------------------------- review counterexamples (§16 revision 2)
def _data_constant_features(seed=0):
    """Domain 2: 5,000 rows where numeric input 2 is ALWAYS 0; domain 3: 3,000 rows where it is 0/1.
    Numeric input 3 is 0 on every training row. The label does not depend on either input."""
    rng = np.random.default_rng(seed)
    dom = np.r_[np.full(5000, 2), np.full(3000, 3)]
    n = dom.size
    cat = np.column_stack([dom, rng.integers(2, 10, n), rng.integers(2, 20, n)]).astype(np.int32)
    num = rng.normal(size=(n, 4)).astype(np.float32)
    num[:, 2] = np.where(dom == 3, rng.integers(0, 2, n), 0)
    num[:, 3] = 0.0
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2.5 + 0.8 * num[:, 0])))).astype(np.float32)
    return MDArrays(cat, num), y


def _fit_constant(variant):
    a, y = _data_constant_features()
    return a, MDModel.fit(a, y, CAT_SIZES, DOM_COL, variant, seed=1, dim=4, epochs=1, batch=1024, lr=3e-3,
                          threads=1)[0]


def test_pn_dimension_constant_in_a_domain_falls_back_to_the_global_statistics():
    a, m = _fit_constant("star")
    net = m.net
    sub = a.take(np.flatnonzero(a.cat[:, DOM_COL] == 2)[:64])
    flipped = MDArrays(sub.cat.copy(), sub.num.copy())
    flipped.num[:, 2] = 1.0  # never seen in domain 2 during training
    cat, num = m.tensors(flipped, slice(None))
    with torch.no_grad():
        dom = net.domain(cat)
        x = net.inputs(cat, num, dom)
        mu, var = net.pn.stats(x, 2, True)
        zn = (x - mu) / torch.sqrt(var + PN_EPS)
        g_mu, g_var = net.pn.global_stats()
    j = net.emb[0].embedding_dim * len(net.in_cols) + 2
    assert torch.allclose(zn[:, j], (x[:, j] - g_mu[j]) / torch.sqrt(g_var[j] + PN_EPS))
    assert float(zn[:, j].abs().max()) < 10  # review: 736 before the fix
    p = m.predict(flipped)
    assert (p > 1e-4).all() and (p < 1 - 1e-4).all()  # review: 0.0586 -> 0.0 before the fix


@pytest.mark.parametrize("variant", ["int", "shared", "head", "star"])
def test_numeric_column_constant_in_training_is_neutral_at_inference(variant):
    a, m = _fit_constant(variant)
    assert m.num_keep[3] == 0 and m.num_keep[:3].all()
    sub = a.take(np.arange(64))
    changed = MDArrays(sub.cat.copy(), sub.num.copy())
    changed.num[:, 3] = 1.0  # review: standardized to 1,000,000 before the fix
    assert np.array_equal(m.predict(changed), m.predict(sub))


def test_star_optimizer_steps_hold_at_least_one_batch_of_rows():
    sizes = {2: 4096, 3: 8191, 4: 5}  # review: these became three single updates of 4,096 / 8,191 / 5 rows
    dom = np.concatenate([np.full(n, d) for d, n in sizes.items()])
    dom_rows = [np.flatnonzero(dom == d) for d in sizes]
    for seed in range(20):
        chunks = domain_batches(np.random.default_rng(seed), dom_rows, 4096)
        chunk_of = np.empty(dom.size, dtype=np.int64)
        for k, c in enumerate(chunks):
            chunk_of[c] = k
        steps = accumulate_steps(chunks, 4096)
        assert np.array_equal(np.sort(np.concatenate(steps)), np.arange(dom.size))
        for st in steps:
            assert 4096 <= st.size < 4 * 4096
            for d in np.unique(dom[st]):  # one chunk per domain per step, so PN statistics stay per chunk
                assert np.unique(chunk_of[st[dom[st] == d]]).size == 1
        tiny = [st for st in steps if (dom[st] == 4).any()]
        assert len(tiny) == 1 and tiny[0].size > 5
    m = _fit("star")[0]
    assert m.info["step_rows_epoch1"]["min"] >= 256
    assert set(m.info["domain_steps_epoch1"]) == set(ROWS)


@pytest.mark.parametrize("variant", ["star", "head"])
def test_single_row_and_batched_predictions_agree_within_the_float32_tolerance(variant):
    m = _fit(variant)[0]
    a, _ = _data(seed=5)
    sub = a.take(np.arange(200))  # mixed domains, so each domain's share differs between the two paths
    batched = m.predict(sub)
    single = np.array([m.predict(sub.take(np.array([i])))[0] for i in range(sub.n)])
    assert np.abs(single - batched).max() <= NN_SINGLE_VS_BATCH_TOL


# ---------------------------------------------------------------- review counterexample (§16 revision 3)
def _two_domain_step(code_a, code_b, seed=0):
    """Domain A (4,096 rows, numeric mean -1) and domain B (512 rows, numeric mean +1) in ONE training step."""
    rng = np.random.default_rng(seed)
    na, nb = 4096, 512
    dom = np.r_[np.full(na, code_a), np.full(nb, code_b)]
    cat = np.column_stack([dom, rng.integers(2, 10, na + nb), rng.integers(2, 20, na + nb)])
    num = np.r_[rng.normal(-1, 1, (na, 4)), rng.normal(1, 1, (nb, 4))].astype(np.float32)
    y = (rng.random(na + nb) < 0.05).astype(np.float32)
    return torch.as_tensor(cat, dtype=torch.long), torch.as_tensor(num), torch.as_tensor(y)


def _star_with_equal_domains():
    torch.manual_seed(0)
    net = MDNet(CAT_SIZES, DOM_COL, 4, "star", dim=4)
    with torch.no_grad():  # codes 2 and 3 start from identical parameters, so only the code differs
        net.emb[DOM_COL].weight[3] = net.emb[DOM_COL].weight[2]
        net.aux_emb.weight[3] = net.aux_emb.weight[2]
    net.seen[2:4] = True
    return net.train()


def test_star_step_does_not_depend_on_which_code_each_domain_has():
    import copy
    base = _star_with_equal_domains()
    nets = {"a2b3": copy.deepcopy(base), "a3b2": copy.deepcopy(base)}
    batches = {"a2b3": _two_domain_step(2, 3), "a3b2": _two_domain_step(3, 2)}
    first = {k: torch.sigmoid(nets[k](*batches[k][:2])).detach() for k in nets}  # training forward = one step
    assert float((first["a2b3"] - first["a3b2"]).abs().max()) < 1e-6  # review: 0.0848 before the fix
    for k in nets:  # the global mean after the first step is the mean of ALL the step's rows
        pooled = batches[k][1].double().mean(0)
        j = slice(net_num_start(nets[k]), None)
        assert torch.allclose(nets[k].pn.glob_mean[j].double(), pooled, atol=1e-5)  # review: 0 instead of -0.7778
    nets = {"a2b3": copy.deepcopy(base), "a3b2": copy.deepcopy(base)}
    loss_fn = nn.BCEWithLogitsLoss()
    for k in nets:
        train_step(nets[k], make_optimizers(nets[k], 1e-2), *batches[k], loss_fn)
        nets[k].eval()
    with torch.no_grad():
        after = {k: torch.sigmoid(nets[k](*batches[k][:2])) for k in nets}
    assert float((after["a2b3"] - after["a3b2"]).abs().max()) < 1e-6  # review: 1.94e-4 before the fix


def net_num_start(net):
    return sum(net.emb[i].embedding_dim for i in net.in_cols)


def test_reload_check_rows_keep_every_prediction_batch_identical():
    from cvr.study_c_worker import PREDICT_BATCH, same_batch_rows
    assert PREDICT_BATCH == 16384
    assert same_batch_rows(50_000, 4_000_000) == 49_152  # 3 full batches, no shorter last batch
    assert same_batch_rows(50_000, 10_000) == 10_000  # one batch either way
    assert same_batch_rows(50_000, 20_000) == 20_000  # the whole split: the same batches as the full pass
    assert same_batch_rows(50_000, 60_000) == 49_152  # a prefix must end on a batch boundary


# ---------------------------------------------------------------- review counterexample (§16 revision 4)
def test_domain_id_embedding_dimensions_always_use_the_global_statistics():
    import copy
    m = _fit("star", epochs=2)[1]
    net, pn = m.net, m.net.pn
    dim = net.emb[DOM_COL].embedding_dim
    start = net.in_cols.index(DOM_COL) * dim
    assert pn.global_dims.sum() == dim and pn.global_dims[start:start + dim].all()
    sub = _rows_with_domain(2, k=300)
    cat, num = m.tensors(sub, slice(None))
    ids = slice(start, start + dim)
    other = slice(start + dim, start + 2 * dim) if start == 0 else slice(0, dim)  # another embedding's dims
    with torch.no_grad():
        g_mu, g_var = pn.global_stats()
        # running statistics that mix the domain-ID embedding of different steps (it drifts while training) can
        # carry a variance far above the 1% rule; the same mix is put on a non-ID block as a control
        for sl in (ids, other):
            pn.dom_msq[2, sl] = pn.dom_mean[2, sl] ** 2 + 4 * g_var[sl]
        x = net.inputs(cat, num, net.domain(cat))
        mu_eval, var_eval = pn.stats(x, 2, True)  # evaluation mode
        trainer = copy.deepcopy(net).train()
        trainer.pn.begin_step()
        mu_tr, var_tr = trainer.pn.stats(x, 2, True)  # training mode, same state and input
    assert torch.equal(mu_eval[other], pn.dom_mean[2, other])  # control: an ordinary dimension uses its own stats
    assert torch.equal(mu_eval[ids], g_mu[ids]) and torch.equal(var_eval[ids], g_var[ids])
    assert torch.equal(mu_tr[ids], g_mu[ids]) and torch.equal(var_tr[ids], g_var[ids])
    z_eval = (x[:, ids] - mu_eval[ids]) / torch.sqrt(var_eval[ids] + PN_EPS)
    z_tr = (x[:, ids] - mu_tr[ids]) / torch.sqrt(var_tr[ids] + PN_EPS)
    assert torch.equal(z_eval, z_tr)  # review: -0.577 (training) vs +1.000 (evaluation) before the rule
