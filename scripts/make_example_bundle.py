"""Build examples/: a small model bundle trained on synthetic MAC-shaped data, a request and its response.

    PYTHONPATH=. uv run python scripts/make_example_bundle.py [--out examples]

The example shows the predictor's interface without downloading MAC (the real models need the 873 MB vocabularies
and a 2 GB checkpoint). Same code path as the benchmark models: synthetic files in MAC's columns -> the official
preprocessing and training (BASE, three files) -> global and group-shrunk Platt fitted on a fourth file -> a bundle.
The official schedule is sized for MAC's 4,000,000 x 21 rows (a 5% warm-up of that), which would leave a model trained on
60,000 rows at the start of its warm-up; the example sizes it for its own rows (--schedule_rows). The data are
synthetic, so the probabilities say nothing about MAC.
"""

import argparse
import json
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from synthetic_mac import make_file, make_world, write_official_vocabs

from cvr import predictor as P
from cvr import pymal_run as R
from cvr.calibration import GlobalPlatt, fit_group_shrunk_exact
from cvr.domain_models import save_domain_calibrators

TRAIN, CALIB = ["train-00.parquet", "train-01.parquet", "train-02.parquet"], "train-03.parquet"
LAMBDA = 100.0  # fixed for the example (the benchmark models choose it on a later file)


def build(out: Path, rows: int = 20000) -> dict:
    import torch

    with tempfile.TemporaryDirectory() as tmp:
        mac = Path(tmp) / "data"
        world = make_world(seed=7, n_users=400, n_items=200)
        for i, name in enumerate([*TRAIN, CALIB]):
            make_file(world, mac / "train" / name, seed=10 + i, rows=rows)
        write_official_vocabs(mac)
        R.preprocess(mac, mac / "cached", [*TRAIN, CALIB])
        a = R.parser().parse_args(
            [
                "train",
                "--mac",
                str(mac),
                "--cache",
                str(mac / "cached"),
                "--out",
                str(Path(tmp) / "run"),
                "--device",
                "cpu",
                "--num_workers",
                "0",
                "--model",
                "BASE",
                "--lr",
                "0.003",
                "--batch_size",
                "256",
                "--seed",
                "2026",
                "--schedule_rows",
                str(rows * len(TRAIN)),
                "--train_files",
                *TRAIN,
                "--ckpt_after",
                TRAIN[-1],
            ]
        )
        run = R.train(a)
        args = json.loads((run / "run.json").read_text())["args"]
        ckpt = run / f"model_after_{Path(TRAIN[-1]).stem}.pt"
        # calibrators on the third file, with the ad groups of the training files
        D, _ = R.official()
        vocabs = D.get_vocabs(str(mac / "vocabs"))
        model, _, _ = R.build(argparse.Namespace(**args), vocabs, torch.device("cpu"))
        model.load_state_dict(torch.load(ckpt))
        model.eval()
        with open(mac / "cached" / CALIB.replace(".parquet", ".pkl"), "rb") as f:
            ds = D.MALDataset(pickle.load(f))
        with torch.no_grad():
            p = (
                model(ds.static_user[:, 1:], ds.static_ad, ds.mm_arr, ds.seq_arr)[args["main_view"]][:, 1]
                .numpy()
                .astype(np.float64)
            )
        y = np.asarray(ds.data["last"])
        train_tokens = set()
        for name in TRAIN:
            train_tokens |= set(
                pq.read_table(mac / "train" / name, columns=["ad_feat1"]).column("ad_feat1").to_pylist()
            )
        codes = {t: i for i, t in enumerate(sorted(train_tokens))}
        calib = pq.read_table(mac / "train" / CALIB)
        g = np.array([codes.get(t, -1) for t in calib.column("ad_feat1").to_pylist()])
        from sklearn.metrics import roc_auc_score

        quality = {
            "auc_on_calibration_file": round(float(roc_auc_score(y, p)), 4),
            "positive_rate": round(float(y.mean()), 4),
            "mean_prediction": round(float(p.mean()), 4),
        }
        glob = GlobalPlatt.fit(p, y)
        gs, _ = fit_group_shrunk_exact(p, y, g, glob, LAMBDA)
        # the bundle
        model_dir = out / "model"
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True)
        shutil.copy(ckpt, model_dir / "model.pt")
        shutil.copytree(mac / "vocabs", model_dir / "vocabs")
        save_domain_calibrators(glob, gs, model_dir / "calibrator.npz")
        (model_dir / "group_codes.json").write_text(json.dumps(codes))
        cfg = {
            "args": args,
            "checkpoint": "model.pt",
            "checkpoint_sha256": R.sha256(model_dir / "model.pt"),
            "calibrator": "calibrator.npz",
            "calibration": "iii",
            "vocabs": "vocabs",
            "source": {
                "data": "synthetic (tests/synthetic_mac.py, seed 7)",
                "trained_on": TRAIN,
                "calibrator_fitted_on": CALIB,
                "lambda": LAMBDA,
            }
            | quality,
        }
        (model_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        # a request: one user of the calibration file with three of that user's rows as candidates
        users = calib.column("user_id").to_pylist()
        counts = {u: users.count(u) for u in set(users)}
        row = next(i for i, u in enumerate(users) if counts[u] >= 3)
        request = P.request_from(mac / "train" / CALIB, row, 3)
    (out / "request.json").write_text(json.dumps(request, indent=1))
    response = P.predict(P.load(model_dir), request)
    (out / "response.json").write_text(json.dumps(response, indent=2) + "\n")
    return response


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "examples")
    print(json.dumps(build(ap.parse_args().out), indent=2))
