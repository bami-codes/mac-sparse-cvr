"""The runner (cvr/pymal_run.py) is the official MAC code with only the additions of docs/DESIGN.md §18.1.

Synthetic MAC-shaped files in the official layout (<path>/data/{train,test,vocabs,cached}), CPU only:
- the official run.py, executed top to bottom (tensors sent to the CPU instead of 'cuda', TensorBoard replaced by a
  recorder because it is not installed), gives the same per-step losses and final parameters as the runner;
- the checkpoint after file k equals a run on files 0..k (the schedule length does not depend on the file count);
- predictions are saved in file row order and equal a direct forward pass; AUC/GAUC agree with this project's code;
- the official test file is scored once per freeze manifest, with every checkpoint and the test file's SHA-256 listed;
- predictions of different checkpoints never replace each other; a stopped run resumed from its last file equals
  the uninterrupted run; the pilot mode keeps no labels or metric; unknown arguments are an error;
- third_party/pymal is byte-identical to the pinned upstream commit.
"""

import argparse
import hashlib
import json
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from synthetic_mac import PAD, make_file, make_world, write_official_vocabs

from cvr import pymal_run as R

TRAIN = ["train-00.parquet", "train-01.parquet", "train-02.parquet"]
MOAE = [
    "--model",
    "MoAE",
    "--lr",
    "0.003",
    "--lamda",
    "0.3",
    "--aux_views",
    "first",
    "mta",
    "linear",
    "cartesian",
    "--batch_size",
    "256",
]


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("pymal")
    mac = root / "data"
    world = make_world(seed=5, n_users=300, n_items=150)
    for i, name in enumerate(TRAIN):
        make_file(world, mac / "train" / name, seed=i, rows=1500)
    make_file(world, mac / "test" / "test.parquet", seed=99, rows=1500)
    write_official_vocabs(mac)
    R.preprocess(mac, mac / "cached", TRAIN + ["test.parquet"])
    yield root, mac
    torch.use_deterministic_algorithms(False)  # the official set_seed switches this on for the whole process


def ours(tree, out: Path, files: list[str], ckpt_after: list[str]) -> Path:
    _, mac = tree
    a = R.parser().parse_args(
        [
            "train",
            "--mac",
            str(mac),
            "--cache",
            str(mac / "cached"),
            "--out",
            str(out),
            "--device",
            "cpu",
            "--num_workers",
            "0",
            *MOAE,
            "--train_files",
            *files,
            "--ckpt_after",
            *ckpt_after,
        ]
    )
    return R.train(a)


def test_the_runner_trains_exactly_like_the_official_run_py(tree, tmp_path, monkeypatch):
    root, _mac = tree
    out = ours(tree, tmp_path / "ours", TRAIN, [TRAIN[-1]])
    our_loss = np.concatenate([np.load(out / f"loss_{Path(n).stem}.npy") for n in TRAIN])
    our_state = torch.load(out / "model_after_train-02.pt")

    _D, _U = R.official()
    import utils.trainer as T

    monkeypatch.setattr(T, "TRAIN_DATA", TRAIN)

    def cpu(args):
        return tuple("cpu" if str(x).startswith("cuda") else x for x in args)

    tensor_to, module_to = torch.Tensor.to, torch.nn.Module.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: tensor_to(self, *cpu(a), **k))
    monkeypatch.setattr(torch.nn.Module, "to", lambda self, *a, **k: module_to(self, *cpu(a), **k))
    scalars = []
    board = types.ModuleType("torch.utils.tensorboard")
    board.SummaryWriter = lambda *a, **k: types.SimpleNamespace(
        add_scalar=lambda tag, v, step: scalars.append((tag, v))
    )
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", board)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--path",
            str(root),
            "--model",
            "MoAE",
            "--lr",
            "0.003",
            "--lamda",
            "0.3",
            "--aux_views",
            "first",
            "mta",
            "linear",
            "cartesian",
            "--batch_size",
            "256",
        ],
    )
    # not run as __main__: macOS DataLoader workers start by spawn and re-run the main module, and run.py has no
    # __main__ guard, so every worker would start the whole training again (the test hung that way)
    g = runpy.run_path(str(R.PYMAL / "run.py"), run_name="pymal_official_run")  # trains, then prints test metrics
    official_loss = np.array([v for tag, v in scalars if tag == "Training/Loss"])
    official_state = g["model"].state_dict()

    assert official_loss.size == our_loss.size > len(TRAIN)  # several steps per file
    assert np.array_equal(official_loss, our_loss)
    assert official_state.keys() == our_state.keys()
    assert all(torch.equal(official_state[k], our_state[k]) for k in our_state)


def test_checkpoint_after_file_k_equals_short_run(tree, tmp_path):
    a = ours(tree, tmp_path / "all", TRAIN, ["train-01.parquet"])
    b = ours(tree, tmp_path / "first_two", TRAIN[:2], ["train-01.parquet"])
    sa, sb = torch.load(a / "model_after_train-01.pt"), torch.load(b / "model_after_train-01.pt")
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    ra, rb = json.loads((a / "run.json").read_text()), json.loads((b / "run.json").read_text())
    assert ra["checkpoints"]["train-01.parquet"]["sha256"] == rb["checkpoints"]["train-01.parquet"]["sha256"]
    assert ra["official_test_evaluated"] is False


def test_predictions_keep_row_order(tree, tmp_path):
    from sklearn.metrics import roc_auc_score

    from cvr.metrics import gauc

    _, mac = tree
    run = ours(tree, tmp_path / "run", TRAIN[:2], ["train-01.parquet"])
    ckpt = run / "model_after_train-01.pt"
    rec = R.predict(predict_args(mac, tmp_path / "pred", [ckpt], "train-02.parquet"))[str(ckpt)]
    z = np.load(tmp_path / "pred" / f"{ckpt.stem}_{R.sha256(ckpt)[:12]}" / "train-02.npz")

    D, _U = R.official()
    import pickle

    with open(mac / "cached" / "train-02.pkl", "rb") as f:
        ds = D.MALDataset(pickle.load(f))
    vocab_user = json.loads((mac / "vocabs" / "user_id.json").read_text())
    raw_users = pq.read_table(mac / "train" / "train-02.parquet").column("user_id").to_pylist()
    assert np.array_equal(z["row"], np.arange(len(raw_users)))
    assert np.array_equal(z["user"], [vocab_user[u] for u in raw_users])  # file row order

    ra = argparse.Namespace(**json.loads((run / "run.json").read_text())["args"])
    model, _, _ = R.build(ra, D.get_vocabs(str(mac / "vocabs")), torch.device("cpu"))
    model.load_state_dict(torch.load(ckpt))
    model.eval()
    with torch.no_grad():
        p = model(ds.static_user[:, 1:], ds.static_ad, ds.mm_arr, ds.seq_arr)["last"][:, 1].numpy()
    assert np.abs(p - z["p_last"]).max() <= 1e-6
    assert set(z.files) >= {"p_first", "p_mta", "p_linear", "y_last"}

    y = z["y_last"]
    m = rec["train-02.parquet"]
    assert abs(m["auc"] - roc_auc_score(y, z["p_last"])) <= 1e-9
    assert abs(m["gauc"] - gauc(y, z["p_last"], z["user"])["gauc"]) <= 1e-9  # official and project GAUC agree


def predict_args(mac: Path, out: Path, ckpts: list[Path], *rest: str) -> argparse.Namespace:
    return R.parser().parse_args(
        [
            "predict",
            "--mac",
            str(mac),
            "--cache",
            str(mac / "cached"),
            "--out",
            str(out),
            "--device",
            "cpu",
            "--num_workers",
            "0",
            *[x for c in ckpts for x in ("--ckpt", str(c))],
            *rest,
        ]
    )


def test_official_test_runs_once_per_manifest(tree, tmp_path):
    _, mac = tree
    run = ours(tree, tmp_path / "run", TRAIN[:2], ["train-00.parquet", "train-01.parquet"])
    c0, c1 = run / "model_after_train-00.pt", run / "model_after_train-01.pt"
    out = tmp_path / "pred"
    with pytest.raises(ValueError, match="freeze manifest"):
        R.predict(predict_args(mac, out, [c0], "test.parquet"))
    with pytest.raises(ValueError, match="bare official names"):  # ./test.parquet must not bypass the manifest
        R.predict(predict_args(mac, out, [c0], "./test.parquet"))
    test_sha = R.sha256(mac / "test" / "test.parquet")
    manifest = tmp_path / "freeze_q2.json"
    manifest.write_text(json.dumps({"checkpoints_sha256": [R.sha256(c0)], "official_test_sha256": test_sha}))
    with pytest.raises(ValueError, match="not listed"):
        R.predict(predict_args(mac, out, [c0, c1], "--official_test", str(manifest), "test.parquet"))
    manifest.write_text(
        json.dumps({"checkpoints_sha256": [R.sha256(c0), R.sha256(c1)], "official_test_sha256": test_sha})
    )
    with pytest.raises(ValueError, match="exactly the 2 frozen checkpoints"):  # a subset must not claim the run
        R.predict(predict_args(mac, out, [c0], "--official_test", str(manifest), "test.parquet"))
    with pytest.raises(ValueError, match="exactly the 2 frozen checkpoints"):  # the same checkpoint twice
        R.predict(predict_args(mac, out, [c0, c0], "--official_test", str(manifest), "test.parquet"))
    manifest.write_text(
        json.dumps({"checkpoints_sha256": [R.sha256(c0), R.sha256(c1)], "official_test_sha256": "0" * 64})
    )
    with pytest.raises(ValueError, match="official test file named"):
        R.predict(predict_args(mac, out, [c0, c1], "--official_test", str(manifest), "test.parquet"))
    assert not list(tmp_path.glob("*.lock"))  # every refusal above leaves the evaluation unused
    manifest.write_text(
        json.dumps({"checkpoints_sha256": [R.sha256(c0), R.sha256(c1)], "official_test_sha256": test_sha})
    )
    R.predict(predict_args(mac, out, [c0, c1], "--official_test", str(manifest), "test.parquet"))  # both seeds at once
    assert json.loads((tmp_path / "freeze_q2.official_test_status.json").read_text())["state"] == "done"
    with pytest.raises(ValueError, match="already used"):  # one evaluation per manifest
        R.predict(predict_args(mac, tmp_path / "pred2", [c0, c1], "--official_test", str(manifest), "test.parquet"))


def test_predictions_of_two_checkpoints_never_replace_each_other(tree, tmp_path):
    _, mac = tree
    run = ours(tree, tmp_path / "run", TRAIN[:2], ["train-00.parquet", "train-01.parquet"])
    c0, c1 = run / "model_after_train-00.pt", run / "model_after_train-01.pt"
    out = tmp_path / "pred"
    rec = R.predict(predict_args(mac, out, [c0], "train-02.parquet"))
    rec |= R.predict(predict_args(mac, out, [c1], "train-02.parquet"))  # must not replace the first file
    files = sorted(out.rglob("train-02.npz"))
    assert len(files) == 2
    for c in (c0, c1):
        r = rec[str(c)]["train-02.parquet"]
        f = out / f"{c.stem}_{R.sha256(c)[:12]}" / "train-02.npz"
        assert R.sha256(f) == r["sha256"]
    with pytest.raises(ValueError, match="never overwritten"):
        R.predict(predict_args(mac, out, [c1], "train-02.parquet"))


def test_resumed_run_equals_uninterrupted_run(tree, tmp_path):
    _, mac = tree
    full = ours(tree, tmp_path / "full", TRAIN, ["train-01.parquet", "train-02.parquet"])
    base = ["train", "--mac", str(mac), "--cache", str(mac / "cached"), "--device", "cpu", "--num_workers", "0"]
    R.train(
        R.parser().parse_args(
            base
            + [
                "--out",
                str(tmp_path / "cut"),
                *MOAE,
                "--train_files",
                *TRAIN,
                "--ckpt_after",
                "train-01.parquet",
                "train-02.parquet",
                "--stop_after",
                "train-01.parquet",
            ]
        )
    )
    cut = tmp_path / "cut"
    assert (cut / "resume.pt").exists() and not (cut / "run.json").exists()
    with pytest.raises(ValueError, match="keeps its training settings"):
        R.train(R.parser().parse_args(base + ["--resume", str(cut), "--lr", "0.004"]))
    R.train(R.parser().parse_args(base + ["--resume", str(cut)]))
    a, b = torch.load(full / "model_after_train-02.pt"), torch.load(cut / "model_after_train-02.pt")
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert np.array_equal(np.load(full / "loss_train-02.npy"), np.load(cut / "loss_train-02.npy"))
    rec = json.loads((cut / "run.json").read_text())
    assert len(rec["sessions"]) == 2 and rec["sessions"][1]["start_file"] == "train-02.parquet"
    assert not (cut / "resume.pt").exists()  # removed when the run finished


def test_the_pilot_mode_keeps_no_labels_or_metric(tree, tmp_path):
    _, mac = tree
    run = ours(tree, tmp_path / "run", TRAIN[:1], ["train-00.parquet"])
    rec = R.predict(
        predict_args(mac, tmp_path / "pred", [run / "model_after_train-00.pt"], "--no_metrics", "train-00.parquet")
    )
    r = next(iter(rec.values()))["train-00.parquet"]
    assert "auc" not in r and r["rows_per_second"] > 0
    z = np.load(next((tmp_path / "pred").rglob("train-00.npz")))
    assert not [f for f in z.files if f.startswith("y_")]


def test_unknown_arguments_are_an_error():
    # the official parser drops them, so the example's --learning_rate 4.0e-3 silently became the default 0.003
    with pytest.raises(SystemExit):
        R.parser().parse_args(["train", "--out", "x", "--lr", "0.003", "--learning_rate", "0.004"])


def test_the_vendored_official_code_is_unmodified():
    blobs = Path(R.PYMAL).parent / "pymal_UPSTREAM_BLOBS.txt"
    want = dict(line.split()[::-1] for line in blobs.read_text().splitlines())
    have = {}
    for p in sorted(Path(R.PYMAL).rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            data = p.read_bytes()
            have[str(p.relative_to(R.PYMAL))] = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    assert have == want


def test_a_diverging_run_stops_and_is_recorded(tree, tmp_path):
    _, mac = tree
    a = R.parser().parse_args(
        [
            "train",
            "--mac",
            str(mac),
            "--cache",
            str(mac / "cached"),
            "--out",
            str(tmp_path / "big"),
            "--device",
            "cpu",
            "--num_workers",
            "0",
            "--model",
            "MoAE",
            "--lr",
            "1e6",
            "--lamda",
            "0.3",
            "--aux_views",
            "first",
            "mta",
            "linear",
            "cartesian",
            "--batch_size",
            "256",
            "--train_files",
            *TRAIN,
            "--ckpt_after",
            TRAIN[-1],
        ]
    )
    out = R.train(a)
    rec = json.loads((out / "run.json").read_text())
    assert rec["diverged"]["step"] >= 1 and not rec["checkpoints"]  # no checkpoint of a diverged run
    assert not (out / "resume.pt").exists()


# ------------------------------------------------------------------ group-weighted main loss
def g1_file(tree, tmp_path) -> Path:
    """A G1 list of five ad_feat1 tokens that occur in the synthetic training files."""
    _, mac = tree
    toks = pq.read_table(mac / "train" / TRAIN[0], columns=["ad_feat1"]).column("ad_feat1").to_pylist()
    g1 = sorted({t for t in toks if t != PAD})[:5]
    f = tmp_path / "group_lists.json"
    f.write_text(json.dumps({"G1": g1, "C": []}))
    return f


def list_sha(groups: Path) -> str:
    import hashlib

    return hashlib.sha256("\n".join(sorted(json.loads(groups.read_text())["G1"])).encode()).hexdigest()


def weighted(tree, out: Path, groups: Path, w: str, files=TRAIN, stop=None, sha=None) -> Path:
    _, mac = tree
    args = [
        "train",
        "--mac",
        str(mac),
        "--cache",
        str(mac / "cached"),
        "--out",
        str(out),
        "--device",
        "cpu",
        "--num_workers",
        "0",
        *MOAE,
        "--train_files",
        *files,
        "--ckpt_after",
        files[-1],
        "--g1_weight",
        w,
        "--weight_groups",
        str(groups),
        "--weight_groups_sha256",
        sha or list_sha(groups),
    ]
    return R.train(R.parser().parse_args(args + (["--stop_after", stop] if stop else [])))


def test_weight_one_is_bit_identical_to_the_unweighted_run(tree, tmp_path):
    groups = g1_file(tree, tmp_path)
    a = ours(tree, tmp_path / "plain", TRAIN, [TRAIN[-1]])
    b = weighted(tree, tmp_path / "w1", groups, "1")
    sa, sb = torch.load(a / "model_after_train-02.pt"), torch.load(b / "model_after_train-02.pt")
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    for n in TRAIN:
        assert np.array_equal(np.load(a / f"loss_{Path(n).stem}.npy"), np.load(b / f"loss_{Path(n).stem}.npy"))


def test_click_weight_normalized_to_mean_one(tree, tmp_path):
    _, mac = tree
    groups = g1_file(tree, tmp_path)
    a = ours(tree, tmp_path / "plain", TRAIN, [TRAIN[-1]])
    b = weighted(tree, tmp_path / "w2", groups, "2")
    sa, sb = torch.load(a / "model_after_train-02.pt"), torch.load(b / "model_after_train-02.pt")
    assert not all(torch.equal(sa[k], sb[k]) for k in sa)
    rec = json.loads((b / "run.json").read_text())["group_weighting"]
    toks = set(json.loads(groups.read_text())["G1"])
    col = [
        t for n in TRAIN for t in pq.read_table(mac / "train" / n, columns=["ad_feat1"]).column("ad_feat1").to_pylist()
    ]
    share = sum(t in toks for t in col) / len(col)
    assert rec["g1_rows"] == sum(t in toks for t in col) and abs(rec["g1_share"] - share) < 1e-15
    assert (
        abs(share * rec["w_g1"] + (1 - share) * rec["w_other"] - 1) < 1e-12
        and abs(rec["w_g1"] / rec["w_other"] - 2) < 1e-12
    )


def test_weighted_loss_weights_each_click():
    D, _ = R.official()
    probs = {"last": torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])}
    labels = {"last": torch.tensor([0, 1, 1, 0])}
    ad = torch.zeros(4, len(D.AD_COLS), dtype=torch.long)
    ad[:, D.AD_COLS.index("ad_feat1")] = torch.tensor([5, 6, 5, 7])
    got = R.weighted_main_loss(probs, labels, "last", ad, (torch.tensor([5]), 1.5, 0.75))
    p = torch.tensor([0.9, 0.8, 0.3, 0.4]) + D.EPSILON
    want = torch.mean(-torch.log(p) * torch.tensor([1.5, 0.75, 1.5, 0.75]))
    assert torch.equal(got, want)
    assert not torch.equal(got, torch.mean(-torch.log(p)) * 1.125)  # not a rescaled batch mean


def test_a_run_recorded_before_amendment_3_still_resumes(tree, tmp_path):
    _, mac = tree
    base = ["train", "--mac", str(mac), "--cache", str(mac / "cached"), "--device", "cpu", "--num_workers", "0"]
    R.train(
        R.parser().parse_args(
            base
            + [
                "--out",
                str(tmp_path / "old"),
                *MOAE,
                "--train_files",
                *TRAIN,
                "--ckpt_after",
                TRAIN[-1],
                "--stop_after",
                TRAIN[0],
            ]
        )
    )
    prog = tmp_path / "old" / "progress.json"
    rec = json.loads(prog.read_text())
    for k in R.NEW_SETTINGS:
        rec["args"].pop(k)  # the grid's runs were recorded without these settings
    prog.write_text(json.dumps(rec))
    R.train(R.parser().parse_args(base + ["--resume", str(tmp_path / "old")]))
    assert (tmp_path / "old" / "run.json").exists()


def test_g1_list_checked_on_start_and_resume(tree, tmp_path):
    _, mac = tree
    groups = g1_file(tree, tmp_path)
    with pytest.raises(ValueError, match="not the fixed"):
        weighted(tree, tmp_path / "wrong", groups, "2", sha="0" * 64)
    out = weighted(tree, tmp_path / "w2", groups, "2", stop=TRAIN[0])
    fixed = list_sha(groups)
    toks = json.loads(groups.read_text())["G1"]
    groups.write_text(json.dumps({"G1": toks[1:], "C": []}))  # another list under the same path
    base = ["train", "--mac", str(mac), "--cache", str(mac / "cached"), "--device", "cpu", "--num_workers", "0"]
    with pytest.raises(ValueError, match="not the fixed"):
        R.train(R.parser().parse_args(base + ["--resume", str(out)]))
    groups.write_text(json.dumps({"G1": toks, "C": []}))
    assert list_sha(groups) == fixed
    R.train(R.parser().parse_args(base + ["--resume", str(out)]))  # the same list: the run continues
    assert json.loads((out / "run.json").read_text())["group_weighting"]["tokens_sha256"] == fixed


# ------------------------------------------------------------------ serving (cvr/serving.py)
def test_serving_rows_give_the_official_tensors_and_predictions(tree, tmp_path, monkeypatch):
    import pickle

    from cvr import evaluation as Q
    from cvr import groups as E
    from cvr import serving as S
    from cvr.calibration import GlobalPlatt, fit_group_shrunk_exact, save_group_calibrators

    _, mac = tree
    run = ours(tree, tmp_path / "run", TRAIN[:1], ["train-00.parquet"])
    ckpt = run / "model_after_train-00.pt"
    D, _ = R.official()
    vocabs = D.get_vocabs(str(mac / "vocabs"))
    # a calibrator for this synthetic run, saved where the study keeps them
    monkeypatch.setattr(E, "OUT", tmp_path / "e_study")
    rid = "MoAE_synthetic"
    (tmp_path / "e_study" / "q2" / "calibrators").mkdir(parents=True)
    table = pq.read_table(mac / "train" / "train-01.parquet")
    toks = sorted(set(table.column("ad_feat1").to_pylist()))
    codes = {t: i for i, t in enumerate(toks)}
    unseen = toks[3]
    codes.pop(unseen)  # this group stands for one absent from the training statistics: its rows are group -1
    with open(mac / "cached" / "train-01.pkl", "rb") as f:
        ds = D.MALDataset(pickle.load(f))
    model, _, _ = R.build(
        argparse.Namespace(**json.loads((run / "run.json").read_text())["args"]), vocabs, torch.device("cpu")
    )
    model.load_state_dict(torch.load(ckpt))
    model.eval()
    with torch.no_grad():
        p_batch = (
            model(ds.static_user[:, 1:], ds.static_ad, ds.mm_arr, ds.seq_arr)["last"][:, 1].numpy().astype(np.float64)
        )
    y = np.asarray(ds.data["last"])
    g = np.array([codes.get(t, -1) for t in table.column("ad_feat1").to_pylist()])
    glob = GlobalPlatt.fit(p_batch, y)
    gs, _ = fit_group_shrunk_exact(p_batch, y, g, glob, 10.0)
    save_group_calibrators(glob, gs, Q.calibrator_path(rid, "iii", "main"))

    server = S.Server(rid, ckpt, vocabs, codes, threads=1)
    rows = table.select(S.Server.columns()).to_pylist()
    idx = [0, 7, 99, 500, len(rows) - 1]
    user, ad, mm, seq = server.tensors([rows[i] for i in idx])
    assert torch.equal(user[:, 1:], ds.static_user[idx, 1:]) and torch.equal(ad, ds.static_ad[idx])  # official mapping
    assert torch.equal(seq, ds.seq_arr[idx]) and torch.equal(mm, ds.mm_arr[idx])  # and layout
    one = np.array([server.raw([rows[i]])[0] for i in idx])  # one row at a time, as served
    assert np.abs(one - p_batch[idx]).max() <= 1e-6
    q = np.array([server.predict([rows[i]])[0] for i in idx])
    assert np.abs(q - gs.predict(p_batch[idx], g[idx])).max() <= 1e-6

    # new inputs: a new user, a value absent from the official vocabulary, a group unseen in training
    r = dict(rows[idx[1]])
    base_q = server.predict([r])[0]
    assert server.predict([r | {"user_id": "never-seen-user"}])[0] == base_q  # user_id is not a model input
    new_ad = r | {"ad_feat0": "never-seen-item"}
    assert server.predict([new_ad])[0] == server.predict([r | {"ad_feat0": PAD}])[0]  # the missing value
    assert server.oov == {"ad_feat0": 1}
    p1 = server.raw([r])
    u_row = r | {"ad_feat1": unseen}
    assert np.allclose(server.calibrate(server.raw([u_row]), [u_row]), gs.predict(server.raw([u_row]), np.array([-1])))
    assert -1 in gs.group_ids  # the pooled unseen group has its own fitted coefficients, not the global mapping
    never = r | {
        "ad_feat1": "never-seen-group"
    }  # also absent from the vocabulary: PAD for the model, -1 for calibration
    assert server.calibrate(p1, [never])[0] == gs.predict(p1, np.array([-1]))[0]

    # one request: the user's features and history encoded once give the same probabilities as one row per candidate
    cands = [rows[i] for i in range(40, 140)]
    owner = rows[7]
    per_row = server.predict([c | {k: owner[k] for k in D.USER_COLS + D.SEQ_COLS} for c in cands])
    shared = server.predict_request(owner, cands)
    assert np.array_equal(per_row, shared)
