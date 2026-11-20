"""M3 feasibility probe on one row group of one training file (not a study result).

Measures id-vocabulary size and coverage for several min_count values, array memory per row,
training time per step on CPU (sparse id embeddings) and MPS (dense), and a rough holdout
LogLoss/AUC for the attention and mean-pooling variants.

Usage:
    PYTHONPATH=. uv run python scripts/m3_probe.py data/raw/mac/train/train-17.parquet runs/m3_probe/train-17_rg0.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from sklearn.metrics import log_loss, roc_auc_score

from cvr import vocab as V
from cvr.features import RAW_COLUMNS, featurize, user_columns
from cvr.nn import ID_COLS, OTHER_CATS, M3Model, count_ids, encode_m3
from cvr.sysinfo import peak_rss_gb


def rss_gb() -> float:
    return peak_rss_gb()


def main(src: str, out: str) -> None:
    t0 = time.time()
    raw = user_columns(pl.from_arrow(pq.ParquetFile(src).read_row_group(0, columns=RAW_COLUMNS)))
    tr_raw = raw.filter(pl.col("bucket") < 70)
    ho_raw = raw.filter(pl.col("bucket") >= 70)
    del raw
    vocab = V.build_vocab(V.count_values([tr_raw.select(V.VOCAB_COLS)]))
    tr, ho = featurize(tr_raw, vocab), featurize(ho_raw, vocab)
    id_counts = count_ids(tr_raw)
    report = {"rows": {"train": tr.height, "holdout": ho.height}, "id_vocab": {}}
    for mc in (1, 2, 5):
        idv = V.build_vocab(id_counts, min_count=mc)
        a_ho = encode_m3(ho_raw, ho, idv)
        report["id_vocab"][f"min_count={mc}"] = {"sizes": V.sizes(idv), "holdout_coverage": a_ho.coverage}
    id_vocab = V.build_vocab(id_counts, min_count=2)
    a_tr, a_ho = encode_m3(tr_raw, tr, id_vocab), encode_m3(ho_raw, ho, id_vocab)
    bytes_per_row = sum(getattr(a_tr, f).nbytes for f in ("other", "target", "seq", "mm", "num")) / a_tr.n
    report["array_bytes_per_row"] = bytes_per_row
    report["prep_seconds"] = round(time.time() - t0, 1)
    other_sizes = [V.sizes(vocab)[c] for c in OTHER_CATS]
    id_sizes = [V.sizes(id_vocab)[c] for c in ID_COLS]
    y_tr, y_ho = tr["y"].to_numpy(), ho["y"].to_numpy()

    # per-step speed: CPU with sparse id embeddings vs MPS with dense ones (first 60 steps)
    speed = {}
    for device in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        sub = np.arange(min(a_tr.n, 60 * 4096))
        from cvr.nn import M3Arrays

        a_sub = M3Arrays(a_tr.other[sub], a_tr.target[sub], a_tr.seq[sub], a_tr.mm[sub], a_tr.num[sub])
        t1 = time.time()
        M3Model.fit(a_sub, y_tr[sub], other_sizes, id_sizes, "attn", seed=0, device=device, threads=4)
        if device == "mps":
            torch.mps.synchronize()
        speed[device] = {
            "steps": int(np.ceil(sub.size / 4096)),
            "seconds": round(time.time() - t1, 2),
            "ms_per_step": round(1000 * (time.time() - t1) / np.ceil(sub.size / 4096), 1),
        }
    report["speed_attn_batch4096"] = speed

    # rough holdout comparison, 1 epoch on CPU, 2 seeds per variant
    res = {}
    for pooling in ("attn", "mean"):
        for seed in (2026, 2027):
            t1 = time.time()
            m = M3Model.fit(a_tr, y_tr, other_sizes, id_sizes, pooling, seed=seed, device="cpu", threads=4)
            p = m.predict(a_ho)
            res[f"{pooling}_s{seed}"] = {
                "logloss": log_loss(y_ho, p),
                "auc": roc_auc_score(y_ho, p),
                "O_over_E": float(y_ho.sum() / p.sum()),
                "seconds": round(time.time() - t1, 1),
                "info": m.info,
            }
    report["holdout_1rowgroup"] = res
    report["note"] = "one row group of one file, users split 70/30; feasibility only, not a study result"
    report["peak_rss_gb"] = round(rss_gb(), 2)
    report["total_seconds"] = round(time.time() - t0, 1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
