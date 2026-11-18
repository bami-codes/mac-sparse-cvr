"""Study E (docs/DESIGN.md §18): train and score the official MAC models (PyMAL) without changing them.

    python -m cvr.pymal_run preprocess --mac data/raw/mac --cache data/pymal_cache train-00.parquet ... test.parquet
    python -m cvr.pymal_run train --mac data/raw/mac --cache data/pymal_cache --out runs/e_pymal/<name> \
        --model MoAE --lr 0.003 --lamda 0.3 --aux_views first mta linear cartesian --seed 2026 --device cuda
    python -m cvr.pymal_run predict --mac data/raw/mac --cache data/pymal_cache \
        --ckpt runs/e_pymal/<name>/model_after_train-18.pt --out <dir> train-19.parquet

Taken from the official code as is (third_party/pymal, commit 3186be8, unmodified): the model classes, the
embedding table, the dataset, the label construction (weight > 0), the loss, set_seed, compute_auc_gauc and the
preprocessing functions. Copied line for line from the official run.py / Trainer.train: the optimizer groups (the
embedding table at 10x the learning rate), the schedule (5% linear warm-up of 4,000,000 x 21 // batch steps, then
constant), the file order and shuffle=False. tests/test_pymal_run.py checks that this loop gives the same losses and
parameters as the official Trainer.train.

Added for the study (§18.1): the device is an argument; unknown arguments are an error (the official parser drops
them, so the example's --learning_rate 4.0e-3 silently became 0.003); checkpoints after the named training files;
no test evaluation inside training; predictions saved in file row order; a run record with the effective settings,
time and memory. The official test file is scored only with a freeze manifest that lists the checkpoint's SHA-256.
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from .sysinfo import cpu_name, peak_rss_gb

PYMAL = Path(__file__).resolve().parents[1] / "third_party" / "pymal"
UPSTREAM_COMMIT = "3186be8fd509ec56e6c537359f8e2a984fc04dfb"
TEST_FILE = "test.parquet"


def official():
    """The official data and utility modules (they import each other as top-level `utils` and `model`)."""
    if str(PYMAL) not in sys.path:
        sys.path.insert(0, str(PYMAL))
    import utils.data as D
    import utils.utils as U
    return D, U


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_file(cache: Path, name: str) -> Path:
    return cache / name.replace(".parquet", ".pkl")


# ------------------------------------------------------------------ preprocessing (official functions)
def preprocess(mac: Path, cache: Path, files: list[str]) -> dict:
    """The official preprocess.py for the named files only (that script runs every file on import). df_to_dict and
    get_vocabs are the official functions; the pickle call is the official save_data_dict."""
    import pandas as pd
    D, _ = official()
    t0 = time.time()
    vocabs = D.get_vocabs(str(mac / "vocabs"))
    rec = {"vocab_seconds": round(time.time() - t0, 1), "vocab_sizes": {c: len(v) for c, v in vocabs.items()},
           "files": {}}
    for name in files:
        t1 = time.time()
        folder = "train" if "train" in name else "test"
        df = pd.read_parquet(mac / folder / name)
        data = D.df_to_dict(df, vocabs)
        out = cache_file(cache, name)
        os.makedirs(out.parent, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        rec["files"][name] = {"rows": int(len(df)), "seconds": round(time.time() - t1, 1),
                              "cache_bytes": out.stat().st_size, "peak_rss_gb_so_far": peak_rss_gb()}
        print(name, rec["files"][name], flush=True)
        del df, data
    rec["peak_rss_gb"] = peak_rss_gb()
    return rec


# ------------------------------------------------------------------ model construction (official run.py)
def build(a, vocabs, device):
    import torch
    _, U = official()
    model = U.get_model(a.model, vocabs, a.hidden_dim, a.main_view, a.aux_views).to(device)
    table = list(model.EmbeddingTable.parameters())
    other = [p for p in model.parameters() if p not in set(table)]
    optimizer = torch.optim.Adam([{"params": table, "lr": a.lr * 10}, {"params": other, "lr": a.lr}])
    total_steps = 4000000 * 21 // a.batch_size
    warmup_steps = int(total_steps * 0.05)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0001, end_factor=1.0, total_iters=warmup_steps)
    constant = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=total_steps - warmup_steps)
    schedule = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, constant], milestones=[warmup_steps])
    return model, optimizer, schedule


# ------------------------------------------------------------------ training (official Trainer.train, device as argument)
def train(a) -> Path:
    import torch
    from torch.utils.data import DataLoader
    D, U = official()
    from utils.grad import get_gcsgrad, get_pcgrad

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)  # a run folder is never reused
    files = a.train_files or list(D.TRAIN_DATA)
    if TEST_FILE in files:
        raise ValueError("the official test file is never a training file")
    missing = [f for f in a.ckpt_after if f not in files]
    if missing:
        raise ValueError(f"checkpoint files {missing} are not training files")
    U.set_seed(a.seed)  # same call order as the official run.py: seed, vocabularies, model, optimizer
    device = torch.device(a.device)
    vocabs = D.get_vocabs(str(Path(a.mac) / "vocabs"))
    model, optimizer, schedule = build(a, vocabs, device)
    all_p = list(model.parameters())
    rec = {"upstream_commit": UPSTREAM_COMMIT, "args": vars(a) | {"train_files": files}, "device": str(device),
           "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else cpu_name(),
           "torch": torch.__version__, "parameters": int(sum(p.numel() for p in all_p)), "files": {},
           "checkpoints": {}, "official_test_evaluated": False}
    steps = 0
    for name in files:
        t0 = time.time()
        model.train()
        with open(cache_file(Path(a.cache), name), "rb") as f:
            data = pickle.load(f)
        dataset = D.MALDataset(data)
        loader = DataLoader(dataset, batch_size=a.batch_size, num_workers=a.num_workers, shuffle=False)
        t_load = time.time() - t0
        losses = []
        for batch in loader:
            user_indices, ad_indices, buy_mm_scores, buy_seq_indices = [batch[x].to(device) for x in ["user", "ad", "mm", "seq"]]
            labels = {k: v.to(device, non_blocking=True) for k, v in batch["label"].items()}
            probs = model(user_indices[:, 1:], ad_indices, buy_mm_scores, buy_seq_indices)
            main_loss, aux_loss_list = U.log_loss(probs, labels, a.main_view, a.aux_views)
            total_loss = main_loss + sum([a.lamda * x for x in aux_loss_list])
            optimizer.zero_grad()
            if a.gcsgrad:
                for p, g in zip(all_p, get_gcsgrad(main_loss, aux_loss_list, all_p)):
                    p.grad = g
            elif a.pcgrad:
                for p, g in zip(all_p, get_pcgrad(main_loss, aux_loss_list, all_p)):
                    p.grad = g
            else:
                total_loss.backward()
            optimizer.step()
            schedule.step()
            losses.append(main_loss.item())  # the official loop logs main_loss.item() every step
            if steps == 0:
                rec["after_first_step"] = memory(device)
            steps += 1
        np.save(out / f"loss_{Path(name).stem}.npy", np.asarray(losses))
        seconds = time.time() - t0
        rec["files"][name] = {"rows": len(dataset), "steps": len(losses), "load_seconds": round(t_load, 1),
                              "seconds": round(seconds, 1), "rows_per_second": round(len(dataset) / max(seconds - t_load, 1e-9)),
                              "mean_main_loss": float(np.mean(losses)) if losses else None} | memory(device)
        if name in a.ckpt_after:
            ckpt = out / f"model_after_{Path(name).stem}.pt"
            torch.save(model.state_dict(), ckpt)
            rec["checkpoints"][name] = {"file": ckpt.name, "sha256": sha256(ckpt), "steps": steps}
        print(name, rec["files"][name], flush=True)
        del loader, dataset, data
    rec["steps"] = steps
    rec["learning_rates_at_end"] = [g["lr"] for g in optimizer.param_groups]  # [embedding table, other]
    (out / "run.json").write_text(json.dumps(rec, indent=2, default=str))
    return out


def memory(device) -> dict:
    import torch
    out = {"peak_rss_gb": peak_rss_gb()}
    if device.type == "cuda":
        out |= {"max_vram_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2),
                "max_vram_reserved_gb": round(torch.cuda.max_memory_reserved(device) / 1e9, 2)}
    return out


# ------------------------------------------------------------------ prediction (official test() without the metrics print)
def predict(a) -> dict:
    import torch
    from torch.utils.data import DataLoader
    D, U = official()
    ckpt = Path(a.ckpt)
    run = json.loads((ckpt.parent / "run.json").read_text())
    ckpt_sha = sha256(ckpt)
    if ckpt_sha not in {c["sha256"] for c in run["checkpoints"].values()}:
        raise ValueError(f"{ckpt} is not a checkpoint recorded in its run.json")
    if TEST_FILE in a.files:
        frozen = json.loads(Path(a.official_test).read_text()) if a.official_test else {}
        if ckpt_sha not in frozen.get("checkpoints_sha256", []):
            raise ValueError("the official test file is scored only with a freeze manifest listing this checkpoint")
    ra = argparse.Namespace(**run["args"])
    device = torch.device(a.device)
    vocabs = D.get_vocabs(str(Path(a.mac) / "vocabs"))
    model, _, _ = build(ra, vocabs, device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    views = sorted(set([ra.main_view] + ra.aux_views) - {"cartesian"})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"checkpoint": str(ckpt), "checkpoint_sha256": ckpt_sha, "main_view": ra.main_view, "files": {}}
    for name in a.files:
        t0 = time.time()
        with open(cache_file(Path(a.cache), name), "rb") as f:
            dataset = D.MALDataset(pickle.load(f))
        loader = DataLoader(dataset, batch_size=a.batch_size, num_workers=a.num_workers, shuffle=False)
        users, probs, labels = [], {v: [] for v in views}, {v: [] for v in views}
        with torch.no_grad():
            for batch in loader:
                user_indices, ad_indices, buy_mm_scores, buy_seq_indices = [batch[x].to(device) for x in ["user", "ad", "mm", "seq"]]
                p = model(user_indices[:, 1:], ad_indices, buy_mm_scores, buy_seq_indices)
                users.append(user_indices[:, 0].cpu().numpy())
                for v in views:
                    probs[v].append(p[v][:, 1].cpu().numpy())
                    labels[v].append(batch["label"][v].numpy())
        users = np.concatenate(users)
        arrays = {"row": np.arange(users.size), "user": users}
        for v in views:
            arrays[f"p_{v}"] = np.concatenate(probs[v])
            arrays[f"y_{v}"] = np.concatenate(labels[v])
        dest = out / f"{Path(name).stem}.npz"
        np.savez(dest, **arrays)
        auc, gauc, n_users = U.compute_auc_gauc(users, arrays[f"y_{ra.main_view}"], arrays[f"p_{ra.main_view}"])
        rec["files"][name] = {"rows": int(users.size), "auc": float(auc), "gauc": float(gauc), "gauc_users": int(n_users),
                              "seconds": round(time.time() - t0, 1), "predictions": dest.name, "sha256": sha256(dest)}
        print(name, rec["files"][name], flush=True)
    (out / f"predict_{ckpt.stem}.json").write_text(json.dumps(rec, indent=2))
    return rec


# ------------------------------------------------------------------ command line (unknown arguments are an error)
def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preprocess", "train", "predict"):
        p = sub.add_parser(name)
        p.add_argument("--mac", type=Path, default=Path("data/raw/mac"))
        p.add_argument("--cache", type=Path, default=Path("data/pymal_cache"))
        if name == "preprocess":
            p.add_argument("files", nargs="+")
            continue
        p.add_argument("--device", default="cuda")
        p.add_argument("--num_workers", type=int, default=8)  # official value
        p.add_argument("--batch_size", type=int, default=4096)
        p.add_argument("--out", required=True)
        if name == "predict":
            p.add_argument("--ckpt", required=True)
            p.add_argument("--official_test", default=None, help="freeze manifest that lists the checkpoint SHA-256")
            p.add_argument("files", nargs="+")
            continue
        p.add_argument("--seed", type=int, default=2026)
        p.add_argument("--model", default="MoAE")
        p.add_argument("--lr", type=float, required=True)  # no default: the value must be visible in the record
        p.add_argument("--hidden_dim", type=int, default=16)
        p.add_argument("--lamda", type=float, default=0.1)
        p.add_argument("--pcgrad", action="store_true")
        p.add_argument("--gcsgrad", action="store_true")
        p.add_argument("--main_view", default="last")
        p.add_argument("--aux_views", nargs="*", default=[])
        p.add_argument("--train_files", nargs="*", default=None, help="default: the official 20 files in order")
        p.add_argument("--ckpt_after", nargs="*", default=["train-17.parquet", "train-18.parquet", "train-19.parquet"])
    return ap


def main(argv: list[str]) -> None:
    a = parser().parse_args(argv)
    if a.cmd == "preprocess":
        rec = preprocess(a.mac, a.cache, a.files)
        a.cache.mkdir(parents=True, exist_ok=True)
        (a.cache / f"preprocess_{int(time.time())}.json").write_text(json.dumps(rec, indent=2))
    elif a.cmd == "train":
        print(f"wrote {train(a)}")
    else:
        predict(a)


if __name__ == "__main__":
    main(sys.argv[1:])
