"""Small synthetic files in the MAC schema, for end-to-end tests that must not need the real data.

What is kept from MAC (cvr/schema.py, docs/DESIGN.md §10.1):
- user_id is a 12-hex string; anonymous hashed categorical columns; the PAD token for missing values;
- an item (ad_feat0) determines its category (ad_feat1), shop (ad_feat2) and ad_feat3..5;
- three purchase-sequence columns of length 20 over the same items (all PAD for users without history) and two
  per-step similarity score sequences;
- a skewed ad-context column (context_feat1) with one dominant value, several values under 1% and two tiny ones;
- four attribution labels (last, first, linear, mta) where mta is positive more often than last.
The label depends on the context group, the item category, purchase-history matches and a user effect, so
models have something to learn. Nothing here reproduces MAC numbers.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PAD = "dbee74ee5fbf"
SEQ = 20
CTX_SHARES = np.array([0.55, 0.12, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.012] + [0.007] * 6 + [0.0004, 0.0002])
CTX_SHARES = CTX_SHARES / CTX_SHARES.sum()


def _tok(prefix: str, i) -> str:
    return hashlib.sha1(f"{prefix}{i}".encode()).hexdigest()[:12]


def make_world(seed: int = 0, n_users: int = 3000, n_items: int = 600) -> dict:
    rng = np.random.default_rng(seed)
    items = {
        "cat": rng.integers(0, 40, n_items),
        "shop": rng.integers(0, 120, n_items),
        "f3": rng.integers(0, 8, n_items),
        "f4": rng.integers(0, 6, n_items),
        "f5": rng.integers(0, 3, n_items),
        "cat_effect": rng.normal(0, 0.7, 40),
    }  # some categories convert rarely (G1 groups of the first study)
    users = {
        "key": rng.choice(16**11, n_users, replace=False) + 16**11,  # 12 hex digits
        "feat": rng.integers(0, 5, (n_users, 6)),
        "effect": rng.normal(0, 0.6, n_users),
        "hist": [
            rng.choice(n_items, rng.integers(0, SEQ + 1)) if rng.random() > 0.05 else np.array([], int)
            for _ in range(n_users)
        ],
    }
    ctx_level = np.r_[-3.0, rng.normal(-3.4, 0.6, CTX_SHARES.size - 1)]
    return {"items": items, "users": users, "ctx_level": ctx_level, "n_items": n_items}


def make_file(world: dict, path: Path, seed: int, rows: int = 20_000) -> Path:
    rng = np.random.default_rng(seed)
    it, us = world["items"], world["users"]
    u = rng.integers(0, us["key"].size, rows)
    item = rng.integers(0, world["n_items"], rows)
    ctx = rng.choice(CTX_SHARES.size, rows, p=CTX_SHARES)
    missing = rng.random(rows) < 0.03  # rows whose item attributes are the PAD token
    seqs, mm0, mm1, match = [], [], [], np.zeros(rows)
    for i in range(rows):
        h = us["hist"][u[i]]
        s = [_tok("i", x) for x in h] + [PAD] * (SEQ - h.size)
        seqs.append(s)
        mm0.append(np.r_[rng.normal(0.1, 0.2, h.size), np.zeros(SEQ - h.size)].astype(float).tolist())
        mm1.append(np.r_[rng.normal(0.0, 0.2, h.size), np.zeros(SEQ - h.size)].astype(float).tolist())
        match[i] = (it["cat"][h] == it["cat"][item[i]]).sum() if h.size else 0
    logit = (
        world["ctx_level"][ctx]
        + it["cat_effect"][it["cat"][item]]
        + 0.35 * np.minimum(match, 3)
        + us["effect"][u]
        + rng.normal(0, 0.3, rows)
    )
    logit[missing] = -9.0
    y = rng.random(rows) < 1 / (1 + np.exp(-logit))
    extra = (~y) & (rng.random(rows) < 0.02) & ~missing
    cols = {
        "user_id": [f"{k:012x}" for k in us["key"][u]],
        "ad_feat0": [_tok("i", x) for x in item],
        **{f"user_feat{j}": [_tok(f"u{j}", x) for x in us["feat"][u, j]] for j in range(6)},
        "context_feat0": [_tok("c0", x) for x in rng.integers(0, 10_000, rows)],
        "context_feat1": [_tok("c1", x) for x in ctx],
        "context_feat2": [_tok("c2", x) for x in rng.integers(0, 12, rows)],
        "context_feat3": [PAD if rng.random() < 0.5 else _tok("c3", x) for x in rng.integers(0, 30, rows)],
        "ad_feat1": [PAD if m else _tok("a1", it["cat"][x]) for m, x in zip(missing, item)],
        "ad_feat2": [PAD if m else _tok("a2", it["shop"][x]) for m, x in zip(missing, item)],
        "ad_feat3": [PAD if m else _tok("a3", it["f3"][x]) for m, x in zip(missing, item)],
        "ad_feat4": [PAD if m else _tok("a4", it["f4"][x]) for m, x in zip(missing, item)],
        "ad_feat5": [PAD if m else _tok("a5", it["f5"][x]) for m, x in zip(missing, item)],
        "ad_feat0_seq": seqs,
        "ad_feat1_seq": [
            [
                PAD if t == PAD else _tok("a1", it["cat"][int(h)])
                for t, h in zip(s, list(us["hist"][uu]) + [0] * (SEQ - len(us["hist"][uu])))
            ]
            for s, uu in zip(seqs, u)
        ],
        "ad_feat2_seq": [
            [
                PAD if t == PAD else _tok("a2", it["shop"][int(h)])
                for t, h in zip(s, list(us["hist"][uu]) + [0] * (SEQ - len(us["hist"][uu])))
            ]
            for s, uu in zip(seqs, u)
        ],
        "mm_feat0_seq": mm0,
        "mm_feat1_seq": mm1,
        "last": (y * rng.integers(1, 4, rows)).astype(float),
        "first": ((y & (rng.random(rows) < 0.9)) * 1.0).astype(float),
        "linear": (y * rng.uniform(0.2, 1.0, rows)).astype(float),
        "mta": ((y | extra) * rng.uniform(0.1, 1.0, rows)).astype(float),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), path, row_group_size=4096)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n")  # like scripts/fetch_mac.py
    return path


def make_mac_tree(root: Path, rows: int = 20_000) -> dict[str, Path]:
    """train-08..14 under root (08-10 training, 11 V1, 12 V2, 13-14 holdout), one shared user/item world."""
    world = make_world()
    return {n: make_file(world, root / f"train-{n:02d}.parquet", seed=n, rows=rows) for n in range(8, 15)}


def write_official_vocabs(mac: Path) -> None:
    """Official vocabulary files ({token: index}, PAD included) over every file under mac/train and mac/test, as the
    official vocabularies cover all of MAC."""
    from cvr import pymal_run as R

    D, _ = R.official()
    tables = [
        pq.read_table(p) for p in sorted((mac / "train").glob("*.parquet")) + sorted((mac / "test").glob("*.parquet"))
    ]
    seq_of = {v: k for k, v in D.MAP_COLS.items()}
    (mac / "vocabs").mkdir(parents=True, exist_ok=True)
    for col in D.USER_COLS + D.AD_COLS:
        values = {PAD}
        for t in tables:
            values |= set(t.column(col).to_pylist())
            if col in seq_of:
                values |= {x for seq in t.column(seq_of[col]).to_pylist() for x in seq}
        (mac / "vocabs" / f"{col}.json").write_text(json.dumps({v: i for i, v in enumerate(sorted(values))}))
