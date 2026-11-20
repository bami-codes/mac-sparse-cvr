r"""Train and score the official MAC models (PyMAL) without changing them.

    python -m cvr.pymal_run preprocess train-00.parquet ... test.parquet
    python -m cvr.pymal_run train --out runs/train/<name> --model MoAE --lr 0.004 --lamda 0.2 \
        --aux_views first mta linear cartesian --seed 2026
    python -m cvr.pymal_run train --resume runs/train/<name>      # continue from the last finished file
    python -m cvr.pymal_run predict --out <dir> --ckpt <run>/model_after_train-18.pt [--ckpt ...] train-19.parquet

The model, dataset, labels, loss, seeding and preprocessing come from third_party/pymal (commit 3186be8) untouched.
The optimizer groups (embedding table at 10x the learning rate), the schedule (5% linear warm-up, then constant), the
file order and shuffle=False are copied line for line from the official run.py. tests/test_pymal_run.py checks that
training here gives the same losses and parameters as running run.py itself.

What the runner adds:
- Unknown arguments are an error and --lr has no default. The official parser silently drops unknown arguments,
  which is how the example's --learning_rate 4.0e-3 ends up training at 0.003.
- Each cache records the SHA-256 of its source file and of itself, checked before use.
- Checkpoints after chosen files, and a resume state after every file, so an interrupted GPU run continues with the
  same result.
- Predictions in file row order, one folder per checkpoint, never overwritten.
- The official test file can only be scored with a freeze manifest that lists every checkpoint, and only once.
- A run record with the effective settings, timings, steady throughput and memory.
"""

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .sysinfo import container_peak_memory_gb, cpu_name, peak_rss_gb

PYMAL = Path(__file__).resolve().parents[1] / "third_party" / "pymal"
UPSTREAM_COMMIT = "3186be8fd509ec56e6c537359f8e2a984fc04dfb"
TEST_FILE = "test.parquet"
NAME = re.compile(r"train-\d{2}\.parquet|test\.parquet")
TIMING_SKIP_STEPS = 50  # steps left out of the steady throughput (worker start-up, first allocations)
TRAINING_SETTINGS = (
    "model",
    "lr",
    "lamda",
    "seed",
    "batch_size",
    "hidden_dim",
    "main_view",
    "aux_views",
    "pcgrad",
    "gcsgrad",
    "train_files",
    "ckpt_after",
    "g1_weight",
    "weight_groups",
    "weight_groups_sha256",
    "schedule_rows",
)
NEW_SETTINGS = ("g1_weight", "weight_groups", "weight_groups_sha256", "schedule_rows")  # absent from older records
OFFICIAL_SCHEDULE_ROWS = 4000000 * 21  # the official run.py's schedule length in rows (about MAC's 21 files)


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


def check_names(files) -> None:
    bad = [f for f in files if not NAME.fullmatch(str(f))]
    if bad:
        raise ValueError(f"file arguments are bare official names (train-NN.parquet, test.parquet), got {bad}")


def source_file(mac: Path, name: str) -> Path:
    return mac / ("test" if name == TEST_FILE else "train") / name


def cache_file(cache: Path, name: str) -> Path:
    return cache / name.replace(".parquet", ".pkl")


def cache_meta(cache: Path, name: str, verify: bool = True) -> dict:
    """The record written with a cache file; with verify, its SHA-256 is checked against the file."""
    pkl = cache_file(cache, name)
    meta = json.loads(pkl.with_name(pkl.name + ".json").read_text())
    if verify and sha256(pkl) != meta["cache_sha256"]:
        raise ValueError(f"{pkl} differs from the cache written by preprocess")
    return meta


def atomic_write(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


# ------------------------------------------------------------------ preprocessing (official functions)
def preprocess(mac: Path, cache: Path, files: list[str]) -> dict:
    """The official preprocess.py for the named files only (that script runs every file on import). df_to_dict and
    get_vocabs are the official functions; the pickle call is the official save_data_dict."""
    import pandas as pd

    check_names(files)
    D, _ = official()
    t0 = time.time()
    vocabs = D.get_vocabs(str(mac / "vocabs"))
    rec = {
        "vocab_seconds": round(time.time() - t0, 1),
        "vocab_sizes": {c: len(v) for c, v in vocabs.items()},
        "files": {},
    }
    for name in files:
        t1 = time.time()
        src = source_file(mac, name)
        df = pd.read_parquet(src)
        data = D.df_to_dict(df, vocabs)
        out = cache_file(cache, name)
        os.makedirs(out.parent, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        meta = {"source": name, "source_sha256": sha256(src), "cache_sha256": sha256(out), "rows": len(df)}
        out.with_name(out.name + ".json").write_text(json.dumps(meta, indent=2))
        rec["files"][name] = meta | {
            "seconds": round(time.time() - t1, 1),
            "cache_bytes": out.stat().st_size,
            "peak_rss_gb_so_far": peak_rss_gb(),
            "container_peak_gb": container_peak_memory_gb(),
        }
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
    # official length, unless the run sizes the schedule for its own data (only the small example model does)
    total_steps = (getattr(a, "schedule_rows", None) or OFFICIAL_SCHEDULE_ROWS) // a.batch_size
    warmup_steps = int(total_steps * 0.05)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0001, end_factor=1.0, total_iters=warmup_steps)
    constant = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=total_steps - warmup_steps)
    schedule = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, constant], milestones=[warmup_steps]
    )
    return model, optimizer, schedule


def group_weights(mac: Path, s, vocabs) -> dict | None:
    """Per-click weights for the main loss: g1_weight on clicks of the fixed G1 groups, 1 elsewhere, scaled so the
    mean over the training rows is 1. The G1 share is recounted from the training files themselves."""
    if not s.g1_weight:
        return None
    import pyarrow as pa
    import pyarrow.compute as pc

    tokens = json.loads(Path(s.weight_groups).read_text())["G1"]
    digest = hashlib.sha256("\n".join(sorted(tokens)).encode()).hexdigest()
    if not s.weight_groups_sha256 or digest != s.weight_groups_sha256:
        raise ValueError(
            f"the G1 list in {s.weight_groups} has SHA-256 {digest[:16]}, not the fixed "
            f"{str(s.weight_groups_sha256)[:16]} (docs/DESIGN.md §18.3)"
        )
    missing = [t for t in tokens if t not in vocabs["ad_feat1"]]
    if missing:
        raise ValueError(f"{len(missing)} G1 tokens are not in the official ad_feat1 vocabulary")
    rows = g1 = 0
    value_set = pa.array(tokens)
    for name in s.train_files:
        col = pq.read_table(source_file(mac, name), columns=["ad_feat1"]).column("ad_feat1")
        rows += len(col)
        g1 += int(pc.sum(pc.is_in(col, value_set=value_set)).as_py() or 0)
    share = g1 / rows
    mean_r = s.g1_weight * share + (1 - share)
    return {
        "g1_weight": s.g1_weight,
        "groups": len(tokens),
        "tokens_sha256": digest,
        "training_rows": rows,
        "g1_rows": g1,
        "g1_share": share,
        "w_g1": s.g1_weight / mean_r,
        "w_other": 1 / mean_r,
        "g1_weight_share": s.g1_weight * share / mean_r,
        "vocab_indices": sorted(vocabs["ad_feat1"][t] for t in tokens),
    }


def weighted_main_loss(probs, labels, main_view, ad_indices, gw):
    """The official main-target loss (utils.log_loss: -mean(log(p_label + EPSILON))) with a weight per click inside
    the mean. With every weight 1 it is bit-identical to the official value (tested)."""
    import torch

    D, _ = official()
    idx, w_g1, w_other = gw
    p_view = probs[main_view]
    prob = p_view[torch.arange(p_view.shape[0], device=p_view.device), labels[main_view]] + D.EPSILON
    in_g1 = torch.isin(ad_indices[:, D.AD_COLS.index("ad_feat1")], idx)
    w = torch.where(in_g1, torch.full_like(prob, w_g1), torch.full_like(prob, w_other))
    return torch.mean(-torch.log(prob) * w)


def memory(device) -> dict:
    import torch

    out = {"peak_rss_gb": peak_rss_gb(), "container_peak_gb": container_peak_memory_gb()}
    if device.type == "cuda":
        out |= {
            "max_vram_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2),
            "max_vram_reserved_gb": round(torch.cuda.max_memory_reserved(device) / 1e9, 2),
        }
    return out


def sync(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ------------------------------------------------------------------ training (official Trainer.train, device as argument)
def train(a) -> Path:
    import torch
    from torch.utils.data import DataLoader

    D, U = official()
    from utils.grad import get_gcsgrad, get_pcgrad

    if not a.resume:
        with_defaults(a)
    if a.resume:
        out = Path(a.resume)
        rec = json.loads((out / "progress.json").read_text())
        saved = rec["args"]
        given = {k: getattr(a, k) for k in TRAINING_SETTINGS if getattr(a, k) is not None}
        changed = {k: (saved.get(k), v) for k, v in given.items() if v != saved.get(k) and k not in ("ckpt_after",)}
        if changed:
            raise ValueError(f"a resumed run keeps its training settings; changed: {changed}")
        s = argparse.Namespace(**{k: saved.get(k) for k in TRAINING_SETTINGS})
    else:
        if a.out is None or a.lr is None:
            raise ValueError("--out and --lr are required for a new run")
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=False)  # a run folder is never reused
        s = argparse.Namespace(**{k: getattr(a, k) for k in TRAINING_SETTINGS})
        s.train_files = s.train_files or list(D.TRAIN_DATA)
        s.ckpt_after = (
            s.ckpt_after if s.ckpt_after is not None else ["train-17.parquet", "train-18.parquet", "train-19.parquet"]
        )
        s.aux_views = s.aux_views or []
        rec = None
    check_names(s.train_files + s.ckpt_after)
    if TEST_FILE in s.train_files:
        raise ValueError("the official test file is never a training file")
    if [f for f in s.ckpt_after if f not in s.train_files]:
        raise ValueError(f"checkpoint files {s.ckpt_after} must be training files")

    U.set_seed(s.seed)  # same call order as the official run.py: seed, vocabularies, model, optimizer
    device = torch.device(a.device)
    vocabs = D.get_vocabs(str(Path(a.mac) / "vocabs"))
    model, optimizer, schedule = build(s, vocabs, device)
    all_p = list(model.parameters())
    weighting = group_weights(Path(a.mac), s, vocabs)
    gw = (
        None
        if weighting is None
        else (
            torch.tensor(weighting["vocab_indices"], device=device),
            float(weighting["w_g1"]),
            float(weighting["w_other"]),
        )
    )
    if a.resume:  # a resumed run must weight the same groups it started with
        now_w = None if weighting is None else {k: v for k, v in weighting.items() if k != "vocab_indices"}
        if now_w != rec.get("group_weighting"):
            raise ValueError(
                f"the group weighting read on resume differs from the run's record: {now_w} vs "
                f"{rec.get('group_weighting')}"
            )
    start, steps = 0, 0
    if rec is None:
        rec = {
            "upstream_commit": UPSTREAM_COMMIT,
            "args": vars(s),
            "parameters": int(sum(p.numel() for p in all_p)),
            "files": {},
            "checkpoints": {},
            "official_test_evaluated": False,
            "sessions": [],
            "group_weighting": None
            if weighting is None
            else {k: v for k, v in weighting.items() if k != "vocab_indices"},
        }
    else:
        state = torch.load(out / "resume.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        torch.set_rng_state(state["rng_cpu"])
        if device.type == "cuda" and state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(state["rng_cuda"], device)
        start, steps = state["next_file"], state["steps"]
    rec["sessions"].append(
        {
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else cpu_name(),
            "start_file": s.train_files[start] if start < len(s.train_files) else None,
        }
    )

    for idx in range(start, len(s.train_files)):
        name = s.train_files[idx]
        t0 = time.time()
        model.train()
        meta = cache_meta(Path(a.cache), name)
        with open(cache_file(Path(a.cache), name), "rb") as f:
            data = pickle.load(f)
        dataset = D.MALDataset(data)
        loader = DataLoader(dataset, batch_size=s.batch_size, num_workers=a.num_workers, shuffle=False)
        t_load = time.time() - t0
        losses, t_steady = [], None
        for batch in loader:
            user_indices, ad_indices, buy_mm_scores, buy_seq_indices = [
                batch[x].to(device) for x in ["user", "ad", "mm", "seq"]
            ]
            labels = {k: v.to(device, non_blocking=True) for k, v in batch["label"].items()}
            probs = model(user_indices[:, 1:], ad_indices, buy_mm_scores, buy_seq_indices)
            main_loss, aux_loss_list = U.log_loss(probs, labels, s.main_view, s.aux_views)
            logged = main_loss  # the official, unweighted main loss is what the loss files record
            if gw is not None:  # weight each click before taking the mean
                main_loss = weighted_main_loss(probs, labels, s.main_view, ad_indices, gw)
            total_loss = main_loss + sum([s.lamda * x for x in aux_loss_list])
            optimizer.zero_grad()
            if s.gcsgrad:
                for p, g in zip(all_p, get_gcsgrad(main_loss, aux_loss_list, all_p)):
                    p.grad = g
            elif s.pcgrad:
                for p, g in zip(all_p, get_pcgrad(main_loss, aux_loss_list, all_p)):
                    p.grad = g
            else:
                total_loss.backward()
            optimizer.step()
            schedule.step()
            losses.append(logged.item())  # the official loop logs main_loss.item() every step (a device sync)
            if steps == 0:
                rec["after_first_step"] = memory(device)
            steps += 1
            if not np.isfinite(losses[-1]):  # diverged: no point paying for the rest
                rec["diverged"] = {"file": name, "step": steps, "loss": str(losses[-1])}
                np.save(out / f"loss_{Path(name).stem}.npy", np.asarray(losses))
                (out / "resume.pt").unlink(missing_ok=True)
                (out / "run.json").write_text(json.dumps(rec, indent=2, default=str))
                print(f"diverged at step {steps} of {name}", flush=True)
                return out
            if len(losses) == TIMING_SKIP_STEPS:
                t_steady = time.time()
        sync(device)
        t_end = time.time()
        np.save(out / f"loss_{Path(name).stem}.npy", np.asarray(losses))
        steady_rows = len(dataset) - TIMING_SKIP_STEPS * s.batch_size
        rec["files"][name] = {
            "rows": len(dataset),
            "source_sha256": meta["source_sha256"],
            "steps": len(losses),
            "load_seconds": round(t_load, 1),
            "seconds": round(t_end - t0, 1),
            "steady_rows_per_second": round(steady_rows / (t_end - t_steady)) if t_steady and steady_rows > 0 else None,
            "mean_main_loss": float(np.mean(losses)) if losses else None,
        } | memory(device)
        if name in s.ckpt_after:
            ckpt = out / f"model_after_{Path(name).stem}.pt"
            atomic_write(ckpt, lambda p: torch.save(model.state_dict(), p))
            rec["checkpoints"][name] = {"file": ckpt.name, "sha256": sha256(ckpt), "steps": steps}
        last = idx == len(s.train_files) - 1
        if not last:
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "schedule": schedule.state_dict(),
                "steps": steps,
                "next_file": idx + 1,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            }
            atomic_write(out / "resume.pt", lambda p, s=state: torch.save(s, p))
        rec["steps"] = steps
        rec["learning_rates_now"] = [g["lr"] for g in optimizer.param_groups]  # [embedding table, other]
        atomic_write(out / "progress.json", lambda p: p.write_text(json.dumps(rec, indent=2, default=str)))
        print(name, rec["files"][name], flush=True)
        del loader, dataset, data
        if a.stop_after == name and not last:  # a planned stop (tests): the run is continued with --resume
            return out
    (out / "resume.pt").unlink(missing_ok=True)  # finished: the resume state is not needed (several GB)
    (out / "run.json").write_text(json.dumps(rec, indent=2, default=str))
    return out


# ------------------------------------------------------------------ prediction (official test() without the metrics print)
def run_record(ckpt: Path) -> dict:
    d = ckpt.parent
    rec = json.loads(((d / "run.json") if (d / "run.json").exists() else (d / "progress.json")).read_text())
    if sha256(ckpt) not in {c["sha256"] for c in rec["checkpoints"].values()}:
        raise ValueError(f"{ckpt} is not a checkpoint recorded by its run")
    return rec


def claim_official_test(manifest_path: Path, ckpts: list[Path], cache: Path) -> tuple[Path, Path]:
    """Every check that can fail without using the evaluation, then the exclusive claim (one per manifest)."""
    manifest = json.loads(manifest_path.read_text())
    listed = list(manifest.get("checkpoints_sha256", []))
    given = [sha256(c) for c in ckpts]
    missing = [str(c) for c, h in zip(ckpts, given) if h not in set(listed)]
    if missing:
        raise ValueError(f"checkpoints not listed in the freeze manifest: {missing}")
    if len(set(given)) != len(given) or sorted(given) != sorted(listed):
        # the whole frozen set, each once: a partial set must not use up the single evaluation
        raise ValueError(
            f"the evaluation needs exactly the {len(listed)} frozen checkpoints, each once; got "
            f"{len(given)} ({len(set(given))} distinct)"
        )
    if cache_meta(cache, TEST_FILE)["source_sha256"] != manifest.get("official_test_sha256"):
        raise ValueError("the cached test file was not made from the official test file named in the manifest")
    lock = manifest_path.with_name(manifest_path.stem + ".official_test.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError(f"{lock} exists: this manifest already used its official test evaluation") from None
    os.write(fd, json.dumps({"pid": os.getpid(), "claimed": time.time()}).encode())
    os.close(fd)
    return lock, manifest_path.with_name(manifest_path.stem + ".official_test_status.json")


def predict(a) -> dict:
    ckpts = [Path(c) for c in a.ckpt]
    check_names(a.files)
    runs = {c: run_record(c) for c in ckpts}
    status = None
    if TEST_FILE in a.files:
        if not a.official_test:
            raise ValueError("the official test file is scored only with a freeze manifest (--official_test)")
        _, status = claim_official_test(Path(a.official_test), ckpts, Path(a.cache))
        status.write_text(json.dumps({"state": "started"}))
    try:
        out = {str(c): score(a, c, runs[c]) for c in ckpts}
    except BaseException as e:
        if status:
            status.write_text(json.dumps({"state": "failed_after_claim", "error": repr(e)[:2000]}))
        raise
    if status:
        status.write_text(json.dumps({"state": "done"}))
    return out


def score(a, ckpt: Path, run: dict) -> dict:
    import torch
    from torch.utils.data import DataLoader

    D, U = official()
    ckpt_sha = sha256(ckpt)
    ra = argparse.Namespace(**run["args"])
    device = torch.device(a.device)
    folder = Path(a.out) / f"{ckpt.stem}_{ckpt_sha[:12]}"  # one folder per checkpoint: no two write the same file
    folder.mkdir(parents=True, exist_ok=True)
    dests = {name: folder / f"{Path(name).stem}.npz" for name in a.files}
    taken = [str(d) for d in dests.values() if d.exists() or d.with_suffix(".json").exists()]
    if taken:
        raise ValueError(f"predictions are never overwritten: {taken}")
    vocabs = D.get_vocabs(str(Path(a.mac) / "vocabs"))
    model, _, _ = build(ra, vocabs, device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    views = sorted(set([ra.main_view] + ra.aux_views) - {"cartesian"})
    rec = {}
    for name in a.files:
        t0 = time.time()
        meta = cache_meta(Path(a.cache), name)
        with open(cache_file(Path(a.cache), name), "rb") as f:
            dataset = D.MALDataset(pickle.load(f))
        loader = DataLoader(dataset, batch_size=a.batch_size, num_workers=a.num_workers, shuffle=False)
        t_load = time.time() - t0
        users, probs, labels = [], {v: [] for v in views}, {v: [] for v in views}
        with torch.no_grad():
            for batch in loader:
                user_indices, ad_indices, buy_mm_scores, buy_seq_indices = [
                    batch[x].to(device) for x in ["user", "ad", "mm", "seq"]
                ]
                p = model(user_indices[:, 1:], ad_indices, buy_mm_scores, buy_seq_indices)
                # copy: a numpy view of a worker's batch keeps its shared-memory file open, and Linux runs out of
                # file descriptors after about 1,000 batches (the official test() copies through the GPU instead)
                users.append(np.array(user_indices[:, 0].cpu().numpy(), copy=True))
                for v in views:
                    probs[v].append(np.array(p[v][:, 1].cpu().numpy(), copy=True))
                    labels[v].append(np.array(batch["label"][v].numpy(), copy=True))
        sync(device)
        seconds = time.time() - t0
        users = np.concatenate(users)
        arrays = {"row": np.arange(users.size), "user": users} | {f"p_{v}": np.concatenate(probs[v]) for v in views}
        r = {
            "rows": int(users.size),
            "source_sha256": meta["source_sha256"],
            "checkpoint_sha256": ckpt_sha,
            "seconds": round(seconds, 1),
            "rows_per_second": round(users.size / max(seconds - t_load, 1e-9)),
        }
        if not a.no_metrics:  # the pilot measures speed only and keeps no labels or metric
            arrays |= {f"y_{v}": np.concatenate(labels[v]) for v in views}
            auc, gauc, n_users = U.compute_auc_gauc(users, arrays[f"y_{ra.main_view}"], arrays[f"p_{ra.main_view}"])
            r |= {"auc": float(auc), "gauc": float(gauc), "gauc_users": int(n_users)}
        np.savez(dests[name], **arrays)
        r["sha256"] = sha256(dests[name])
        dests[name].with_suffix(".json").write_text(json.dumps(r, indent=2))
        rec[name] = r
        print(ckpt.name, name, r, flush=True)
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
        if name == "predict":
            p.add_argument("--out", required=True)
            p.add_argument("--batch_size", type=int, default=4096)
            p.add_argument(
                "--ckpt",
                action="append",
                required=True,  # repeated, so it cannot swallow the file names
                help="a checkpoint; repeat the flag for several",
            )
            p.add_argument("--official_test", default=None, help="freeze manifest: checkpoints and test SHA-256")
            p.add_argument("--no_metrics", action="store_true", help="speed only: no labels or metric are kept")
            p.add_argument("files", nargs="+")
            continue
        p.add_argument("--out", default=None)
        p.add_argument("--resume", default=None, help="continue a stopped run folder from its last finished file")
        p.add_argument("--stop_after", default=None, help="stop after this file (tests of --resume)")
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--model", default=None)
        p.add_argument("--lr", type=float, default=None)  # no default: the value must be visible in the record
        p.add_argument("--hidden_dim", type=int, default=None)
        p.add_argument("--lamda", type=float, default=None)
        p.add_argument("--batch_size", type=int, default=None)
        p.add_argument("--pcgrad", action="store_true", default=None)
        p.add_argument("--gcsgrad", action="store_true", default=None)
        p.add_argument("--main_view", default=None)
        p.add_argument("--aux_views", nargs="*", default=None)
        p.add_argument("--train_files", nargs="*", default=None, help="default: the official 20 files in order")
        p.add_argument("--ckpt_after", nargs="*", default=None, help="default: train-17, train-18, train-19")
        p.add_argument("--g1_weight", type=float, default=None, help="weight of the G1 clicks in the main loss")
        p.add_argument("--weight_groups", default=None, help="group_lists.json holding the fixed G1 tokens")
        p.add_argument("--weight_groups_sha256", default=None, help="the fixed G1 list's SHA-256 (groups.json)")
        p.add_argument(
            "--schedule_rows",
            type=int,
            default=None,
            help="rows the warm-up and schedule are sized for (default: the official 4,000,000 x 21)",
        )
    return ap


DEFAULTS = {
    "seed": 2026,
    "model": "MoAE",
    "hidden_dim": 16,
    "lamda": 0.1,
    "batch_size": 4096,
    "pcgrad": False,
    "gcsgrad": False,
    "main_view": "last",
}  # the official defaults, applied to new runs only


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


def with_defaults(a):
    if not a.resume:
        for k, v in DEFAULTS.items():
            if getattr(a, k) is None:
                setattr(a, k, v)
    return a


if __name__ == "__main__":
    main(sys.argv[1:])
