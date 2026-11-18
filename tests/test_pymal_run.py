"""The study E runner (cvr/pymal_run.py) is the official MAC code with only the additions of docs/DESIGN.md §18.1.

Synthetic MAC-shaped files in the official layout (<path>/data/{train,test,vocabs,cached}), CPU only:
- the official run.py, executed top to bottom (tensors sent to the CPU instead of 'cuda', TensorBoard replaced by a
  recorder because it is not installed), gives the same per-step losses and final parameters as the runner;
- the checkpoint after file k equals a run on files 0..k (the schedule length does not depend on the file count);
- predictions are saved in file row order and equal a direct forward pass; AUC/GAUC agree with this project's code;
- the official test file is refused without a freeze manifest; unknown arguments are an error;
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
from synthetic_mac import PAD, make_file, make_world  # noqa: E402

from cvr import pymal_run as R  # noqa: E402

TRAIN = ["train-00.parquet", "train-01.parquet", "train-02.parquet"]
MOAE = ["--model", "MoAE", "--lr", "0.003", "--lamda", "0.3", "--aux_views", "first", "mta", "linear", "cartesian",
        "--batch_size", "256"]


def write_vocabs(mac: Path) -> None:
    """Official format {token: index}, every value of every file (the official vocabularies cover all of MAC)."""
    D, _ = R.official()
    tables = [pq.read_table(p) for p in sorted((mac / "train").glob("*.parquet")) + [mac / "test" / "test.parquet"]]
    seq_of = {v: k for k, v in D.MAP_COLS.items()}
    (mac / "vocabs").mkdir(parents=True, exist_ok=True)
    for col in D.USER_COLS + D.AD_COLS:
        values = {PAD}
        for t in tables:
            values |= set(t.column(col).to_pylist())
            if col in seq_of:
                values |= {x for seq in t.column(seq_of[col]).to_pylist() for x in seq}
        (mac / "vocabs" / f"{col}.json").write_text(json.dumps({v: i for i, v in enumerate(sorted(values))}))


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("pymal")
    mac = root / "data"
    world = make_world(seed=5, n_users=300, n_items=150)
    for i, name in enumerate(TRAIN):
        make_file(world, mac / "train" / name, seed=i, rows=1500)
    make_file(world, mac / "test" / "test.parquet", seed=99, rows=1500)
    write_vocabs(mac)
    R.preprocess(mac, mac / "cached", TRAIN + ["test.parquet"])
    yield root, mac
    torch.use_deterministic_algorithms(False)  # the official set_seed switches this on for the whole process


def ours(tree, out: Path, files: list[str], ckpt_after: list[str]) -> Path:
    _, mac = tree
    a = R.parser().parse_args(["train", "--mac", str(mac), "--cache", str(mac / "cached"), "--out", str(out),
                               "--device", "cpu", "--num_workers", "0", *MOAE, "--train_files", *files,
                               "--ckpt_after", *ckpt_after])
    return R.train(a)


def test_the_runner_trains_exactly_like_the_official_run_py(tree, tmp_path, monkeypatch):
    root, mac = tree
    out = ours(tree, tmp_path / "ours", TRAIN, [TRAIN[-1]])
    our_loss = np.concatenate([np.load(out / f"loss_{Path(n).stem}.npy") for n in TRAIN])
    our_state = torch.load(out / "model_after_train-02.pt")

    D, U = R.official()
    import utils.trainer as T
    monkeypatch.setattr(T, "TRAIN_DATA", TRAIN)

    def cpu(args):
        return tuple("cpu" if str(x).startswith("cuda") else x for x in args)
    tensor_to, module_to = torch.Tensor.to, torch.nn.Module.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: tensor_to(self, *cpu(a), **k))
    monkeypatch.setattr(torch.nn.Module, "to", lambda self, *a, **k: module_to(self, *cpu(a), **k))
    scalars = []
    board = types.ModuleType("torch.utils.tensorboard")
    board.SummaryWriter = lambda *a, **k: types.SimpleNamespace(add_scalar=lambda tag, v, step: scalars.append((tag, v)))
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", board)
    monkeypatch.setattr(sys, "argv", ["run.py", "--path", str(root), "--model", "MoAE", "--lr", "0.003",
                                      "--lamda", "0.3", "--aux_views", "first", "mta", "linear", "cartesian",
                                      "--batch_size", "256"])
    # not run as __main__: macOS DataLoader workers start by spawn and re-run the main module, and run.py has no
    # __main__ guard, so every worker would start the whole training again (the test hung that way)
    g = runpy.run_path(str(R.PYMAL / "run.py"), run_name="pymal_official_run")  # trains, then prints test metrics
    official_loss = np.array([v for tag, v in scalars if tag == "Training/Loss"])
    official_state = g["model"].state_dict()

    assert official_loss.size == our_loss.size > len(TRAIN)  # several steps per file
    assert np.array_equal(official_loss, our_loss)
    assert official_state.keys() == our_state.keys()
    assert all(torch.equal(official_state[k], our_state[k]) for k in our_state)


def test_the_checkpoint_after_file_k_equals_a_run_on_files_0_to_k(tree, tmp_path):
    a = ours(tree, tmp_path / "all", TRAIN, ["train-01.parquet"])
    b = ours(tree, tmp_path / "first_two", TRAIN[:2], ["train-01.parquet"])
    sa, sb = torch.load(a / "model_after_train-01.pt"), torch.load(b / "model_after_train-01.pt")
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    ra, rb = json.loads((a / "run.json").read_text()), json.loads((b / "run.json").read_text())
    assert ra["checkpoints"]["train-01.parquet"]["sha256"] == rb["checkpoints"]["train-01.parquet"]["sha256"]
    assert ra["official_test_evaluated"] is False


def test_predictions_keep_the_row_order_and_match_a_direct_forward_pass(tree, tmp_path):
    from sklearn.metrics import roc_auc_score

    from cvr.metrics import gauc
    _, mac = tree
    run = ours(tree, tmp_path / "run", TRAIN[:2], ["train-01.parquet"])
    ckpt = run / "model_after_train-01.pt"
    a = R.parser().parse_args(["predict", "--mac", str(mac), "--cache", str(mac / "cached"), "--ckpt", str(ckpt),
                               "--out", str(tmp_path / "pred"), "--device", "cpu", "--num_workers", "0",
                               "train-02.parquet"])
    rec = R.predict(a)
    z = np.load(tmp_path / "pred" / "train-02.npz")

    D, U = R.official()
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
    m = rec["files"]["train-02.parquet"]
    assert abs(m["auc"] - roc_auc_score(y, z["p_last"])) <= 1e-9
    assert abs(m["gauc"] - gauc(y, z["p_last"], z["user"])["gauc"]) <= 1e-9  # official and project GAUC agree


def test_the_official_test_file_needs_a_freeze_manifest(tree, tmp_path):
    _, mac = tree
    run = ours(tree, tmp_path / "run", TRAIN[:1], ["train-00.parquet"])
    ckpt = run / "model_after_train-00.pt"
    base = ["predict", "--mac", str(mac), "--cache", str(mac / "cached"), "--ckpt", str(ckpt), "--out",
            str(tmp_path / "pred"), "--device", "cpu", "--num_workers", "0"]
    with pytest.raises(ValueError, match="freeze manifest"):
        R.predict(R.parser().parse_args(base + ["test.parquet"]))
    manifest = tmp_path / "freeze.json"
    manifest.write_text(json.dumps({"checkpoints_sha256": [R.sha256(ckpt)]}))
    R.predict(R.parser().parse_args(base + ["--official_test", str(manifest), "test.parquet"]))
    assert (tmp_path / "pred" / "test.npz").exists()


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
