"""M3 id vocabulary and encoding on synthetic raw MAC-like rows (no real data needed).

Checks that (1) the id vocabulary comes from TRAINING rows only, (2) an item that appears only
in validation becomes UNKNOWN (a real purchase, kept as a valid step) while PAD becomes MISSING
(masked), and (3) the network attends to unknown steps but ignores PAD steps.
"""

import polars as pl
import torch

from cvr.schema import MISSING, PAD, UNKNOWN
from experiments import vocab as V
from experiments.features import RAW_COLUMNS, featurize, user_columns
from experiments.m3data import ID_COLS, OTHER_CATS, count_ids, encode_m3
from experiments.nn import M3Net

L = 20


def make_raw(rows: list[dict]) -> pl.DataFrame:
    """rows: dicts with user, item, cat, shop, seq (list of (item, cat, shop) or PAD), mm0, last."""
    out = {c: [] for c in RAW_COLUMNS}
    for r in rows:
        seq = r["seq"] + [PAD] * (L - len(r["seq"]))
        out["user_id"].append(r["user"])
        out["ad_feat0"].append(r["item"])
        out["ad_feat1"].append(r["cat"])
        out["ad_feat2"].append(r["shop"])
        for c in ["ad_feat3", "ad_feat4", "ad_feat5", "context_feat1", "context_feat2", "context_feat3"] + [
            f"user_feat{i}" for i in range(6)
        ]:
            out[c].append("v1")
        for j, c in enumerate(["ad_feat0_seq", "ad_feat1_seq", "ad_feat2_seq"]):
            out[c].append([s if s == PAD else s[j] for s in seq])
        out["mm_feat0_seq"].append(r.get("mm0", [0.1] * L))
        out["mm_feat1_seq"].append([0.2] * L)
        for c in ["last", "first", "linear", "mta"]:
            out[c].append(float(r.get("last", 0.0)))
    return user_columns(pl.DataFrame(out))


TRAIN = [
    {"user": "00000000000a", "item": "i1", "cat": "c1", "shop": "s1", "seq": [("i1", "c1", "s1"), ("i2", "c1", "s2")]},
    {"user": "00000000000b", "item": "i2", "cat": "c1", "shop": "s2", "seq": [("i2", "c1", "s2")], "last": 1.0},
    {"user": "00000000000c", "item": "i1", "cat": "c2", "shop": "s1", "seq": []},  # empty history
]
VALID = [  # "i9" / "c9" / "s9" appear only here
    {"user": "00000000000d", "item": "i9", "cat": "c1", "shop": "s9", "seq": [("i9", "c9", "s9"), ("i1", "c1", "s1")]},
]


def test_id_vocabulary_from_training_rows_only():
    train, valid = make_raw(TRAIN), make_raw(VALID)
    id_vocab = V.build_vocab(count_ids(train), min_count=1)
    assert "i9" not in id_vocab["ad_feat0"]["value"].to_list()
    assert "s9" not in id_vocab["ad_feat2"]["value"].to_list()
    vocab = V.build_vocab(V.count_values([train.select(V.VOCAB_COLS)]))
    a = encode_m3(valid, featurize(valid, vocab), id_vocab)
    code = dict(zip(id_vocab["ad_feat0"]["value"], id_vocab["ad_feat0"]["code"]))
    assert a.target[0, 0] == UNKNOWN  # validation-only target item
    assert a.seq[0, 0, 0] == UNKNOWN  # validation-only purchased item -> UNKNOWN, not PAD
    assert a.seq[0, 1, 0] == code["i1"]
    assert (a.seq[0, 2:, :] == MISSING).all()  # padding stays MISSING in all three id columns
    assert a.coverage["ad_feat0"]["seq_unknown_share_of_nonpad"] == 0.5


def test_network_attends_to_unknown_steps_but_ignores_pad():
    train, valid = make_raw(TRAIN), make_raw(VALID)
    id_vocab = V.build_vocab(count_ids(train), min_count=1)
    vocab = V.build_vocab(V.count_values([train.select(V.VOCAB_COLS)]))
    a = encode_m3(valid, featurize(valid, vocab), id_vocab)
    other_sizes = [V.sizes(vocab)[c] for c in OTHER_CATS]
    id_sizes = [V.sizes(id_vocab)[c] for c in ID_COLS]
    for pooling in ("attn", "mean"):
        torch.manual_seed(0)
        net = M3Net(other_sizes, id_sizes, n_num=a.num.shape[1], dim=4, pooling=pooling, sparse=False).eval()

        def run(mm):
            t = [torch.as_tensor(x, dtype=torch.long) for x in (a.other, a.target, a.seq)]
            return net(*t, torch.as_tensor(mm), torch.as_tensor(a.num)).item()

        base = run(a.mm)
        mm_unknown = a.mm.copy()
        mm_unknown[0, 0, :] += 5.0  # step 0 holds the UNKNOWN item: a real purchase, must matter
        mm_pad = a.mm.copy()
        mm_pad[0, 5:, :] += 5.0  # PAD steps: must not matter
        assert abs(run(mm_unknown) - base) > 1e-6
        assert abs(run(mm_pad) - base) < 1e-7
