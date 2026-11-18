# Official MAC code (PyMAL), unmodified

- Source: https://github.com/alimama-tech/PyMAL, commit `3186be8fd509ec56e6c537359f8e2a984fc04dfb` (2026-03-03, "update Arxiv links").
- License: Apache-2.0 ([pymal/LICENSE](pymal/LICENSE)). Copyright belongs to the original authors.
- `third_party/pymal/` is a copy of that commit without `assets/` (one figure). No file is changed;
  [pymal_UPSTREAM_BLOBS.txt](pymal_UPSTREAM_BLOBS.txt) lists the git blob id of every file at that commit and
  `tests/test_pymal_run.py` checks the copy against it.
- This project's runner ([cvr/pymal_run.py](../cvr/pymal_run.py)) imports the official model and data code as is and
  adds only what the study needs (docs/DESIGN.md §18.1): device choice, checkpoints at file boundaries, no test
  evaluation inside training, prediction files in row order, and a run record.
