"""Row sampling and positional reading of the study C serving check (scripts/latency_c.py)."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.latency_c import RAW, read_rows, sample_indices


def test_sample_covers_every_domain_and_is_sorted_unique():
    rng = np.random.default_rng(0)
    dom = rng.choice(np.arange(2, 40), size=50_000, p=np.r_[0.6, np.full(37, 0.4 / 37)])
    dom[123] = 77  # a domain with a single row must still be sampled
    idx = sample_indices(dom)
    assert set(np.unique(dom[idx])) == set(np.unique(dom))
    assert np.array_equal(idx, np.unique(idx))


def test_rows_are_read_at_their_file_positions(tmp_path):
    n = 1000
    cols = {c: [f"{c}_{i}" for i in range(n)] for c in RAW if not c.endswith("_seq")}
    cols |= {c: [[f"{c}_{i}"] * 3 for i in range(n)] for c in RAW if c.endswith("_seq")}
    f = tmp_path / "t.parquet"
    pq.write_table(pa.table(cols), f, row_group_size=128)  # 8 row groups
    idx = np.array([0, 5, 127, 128, 400, 999])
    rows = read_rows(str(f), idx)
    assert [r["user_id"] for r in rows] == [f"user_id_{i}" for i in idx]
