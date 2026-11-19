"""The predictor (cvr/predictor.py) and its example (examples/): a request in, calibrated probabilities out."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cvr import predictor as P

ROOT = Path(__file__).resolve().parents[1]
REQUEST = json.loads((ROOT / "examples" / "request.json").read_text())


@pytest.fixture(scope="module")
def server():
    return P.load()


def test_the_example_answers_its_saved_response(server):
    got = P.predict(server, REQUEST)
    want = json.loads((ROOT / "examples" / "response.json").read_text())
    assert [c["ad_feat0"] for c in got["candidates"]] == [c["ad_feat0"] for c in want["candidates"]]
    for key in ("cvr", "cvr_uncalibrated"):
        assert np.allclose([c[key] for c in got["candidates"]], [c[key] for c in want["candidates"]], atol=1e-6)
    assert got["values_not_in_vocabulary"] == {}


def test_the_response_is_the_server_request_path(server):
    got = P.predict(server, REQUEST)
    direct = server.predict_request(REQUEST["user"], REQUEST["candidates"])
    assert np.allclose([c["cvr"] for c in got["candidates"]], direct, atol=1e-8)
    per_row = server.predict([c | REQUEST["user"] for c in REQUEST["candidates"]])  # the same as one row per candidate
    assert np.allclose(direct, per_row, atol=1e-7)


def test_the_command_line(tmp_path):
    out = subprocess.run([sys.executable, "-m", "cvr.predictor", str(ROOT / "examples" / "request.json")],
                         capture_output=True, text=True, cwd=ROOT, check=True)
    assert len(json.loads(out.stdout)["candidates"]) == 3
    bad = copy.deepcopy(REQUEST)
    del bad["user"]["user_feat0"]
    lines = "\n".join([json.dumps(REQUEST), json.dumps(bad), json.dumps(REQUEST)])
    out = subprocess.run([sys.executable, "-m", "cvr.predictor", "-", "--jsonl"], input=lines,
                         capture_output=True, text=True, cwd=ROOT, check=True)
    answers = [json.loads(x) for x in out.stdout.splitlines()]
    assert len(answers) == 3 and "error" in answers[1] and answers[0] == answers[2]


@pytest.mark.parametrize("break_it, message", [
    (lambda r: r["user"].pop("ad_feat0_seq"), "user lacks"),
    (lambda r: r["user"]["ad_feat1_seq"].pop(), "one length"),
    (lambda r: r["candidates"][1].pop("context_feat2"), "candidate 1 lacks"),
    (lambda r: r["candidates"][0]["mm_feat0_seq"].pop(), "one similarity per history item"),
    (lambda r: r.update(candidates=[]), "non-empty"),
])
def test_invalid_requests_are_refused_with_the_reason(server, break_it, message):
    r = copy.deepcopy(REQUEST)
    break_it(r)
    with pytest.raises(P.RequestError, match=message):
        P.predict(server, r)


def test_new_values_are_served_and_counted(server):
    r = copy.deepcopy(REQUEST)
    r["candidates"][0]["ad_feat0"] = "never-seen-ad"
    r["candidates"][0]["ad_feat1"] = "never-seen-group"
    r["user"]["user_id"] = "never-seen-user"
    got = P.predict(server, r)
    assert got["values_not_in_vocabulary"] == {"ad_feat0": 1, "ad_feat1": 1}
    assert all(0 < c["cvr"] < 1 for c in got["candidates"])


def test_a_bundle_refuses_another_checkpoint(tmp_path):
    import shutil
    b = tmp_path / "bundle"
    shutil.copytree(P.EXAMPLE, b)
    cfg = json.loads((b / "config.json").read_text())
    (b / "config.json").write_text(json.dumps(cfg | {"checkpoint_sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="not the checkpoint"):
        P.load(b)


def test_the_example_is_rebuilt_identically(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_example_bundle as M
    import torch
    try:
        rebuilt = M.build(tmp_path)
    finally:
        torch.use_deterministic_algorithms(False)  # the official set_seed switches it on for the whole process
    want = json.loads((ROOT / "examples" / "response.json").read_text())
    assert np.allclose([c["cvr"] for c in rebuilt["candidates"]], [c["cvr"] for c in want["candidates"]], atol=1e-6)
