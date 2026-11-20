"""Score one user's ad candidates and return a calibrated CVR for each.

    uv run python -m cvr.predictor examples/request.json
    uv run python -m cvr.predictor request.json --bundle bundles/BASE_train17 --vocabs data/raw/mac/vocabs
    uv run python -m cvr.predictor - --jsonl < requests.jsonl      # keep the model loaded, one request per line

    uv run python -m cvr.predictor export BASE_lr0.003_lam0.3_s2026 bundles/BASE_train17
    uv run python -m cvr.predictor request-from data/raw/mac/train/train-19.parquet 123 > request.json

A model ships as a folder (bundle):
    config.json        training args of the official model, checkpoint path and SHA-256
    calibrator.npz     global and group-shrunk Platt, fitted on this checkpoint's predictions (optional)
    group_codes.json   ad group (ad_feat1) token -> calibrator group; unknown tokens fall into the "unseen" group

A request uses MAC's raw columns:
    {"user": {user_feat0..5, ad_feat0_seq..ad_feat2_seq},
     "candidates": [{ad_feat0..5, context_feat0..3, mm_feat0_seq, mm_feat1_seq}, ...]}
The three history lists hold MAC's 20 slots (padded), and each candidate carries one similarity per slot.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from . import pymal_run
from .serving import Server

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "model"
CALIBRATIONS = {"iii": "group-shrunk Platt on the ad group (ad_feat1)", "ii": "global Platt", "none": "none"}


class RequestError(ValueError):
    pass


def columns() -> tuple[list[str], list[str]]:
    D, _ = pymal_run.official()
    return D.USER_COLS[1:] + D.SEQ_COLS, D.AD_COLS + D.MM_COLS


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate(request: dict) -> None:
    """Raise RequestError naming the first field that is missing or has the wrong type or length."""
    D, _ = pymal_run.official()
    if not isinstance(request, dict) or not isinstance(request.get("user"), dict):
        raise RequestError('a request is {"user": {...}, "candidates": [{...}, ...]}')
    cands = request.get("candidates")
    if not isinstance(cands, list) or not cands:
        raise RequestError("candidates must be a non-empty list")
    user = request["user"]
    if missing := [c for c in D.USER_COLS[1:] + D.SEQ_COLS if c not in user]:
        raise RequestError(f"user lacks {missing}")
    if bad := [c for c in D.USER_COLS[1:] if not isinstance(user[c], str)]:
        raise RequestError(f"user {bad} must be strings (MAC tokens)")
    if bad := [c for c in D.SEQ_COLS if not (isinstance(user[c], list) and all(isinstance(t, str) for t in user[c]))]:
        raise RequestError(f"user {bad} must be lists of strings")
    # MAC always stores the last SEQ_LEN purchases, padded with the missing token. A shorter list is not "no history":
    # the model would see fewer slots than it was trained on and give another probability, so it is refused.
    lengths = {c: len(user[c]) for c in D.SEQ_COLS}
    n = D.SEQ_LEN
    if set(lengths.values()) != {n}:
        raise RequestError(f"the purchase history columns must each hold {n} items, padded as in MAC: {lengths}")
    for i, c in enumerate(cands):
        if not isinstance(c, dict):
            raise RequestError(f"candidate {i} must be an object")
        if missing := [k for k in D.AD_COLS + D.MM_COLS if k not in c]:
            raise RequestError(f"candidate {i} lacks {missing}")
        if bad := [k for k in D.AD_COLS if not isinstance(c[k], str)]:
            raise RequestError(f"candidate {i}: {bad} must be strings (MAC tokens)")
        bad = [k for k in D.MM_COLS if not (isinstance(c[k], list) and len(c[k]) == n and all(map(_number, c[k])))]
        if bad:
            raise RequestError(f"candidate {i}: {bad} must hold one number per history item ({n})")


def load(bundle: Path = EXAMPLE, vocabs: Path | None = None, threads: int = 1) -> Server:
    """Load a bundle. Real bundles keep the 873 MB vocabularies outside, so pass them with `vocabs`."""
    bundle = Path(bundle)
    cfg = json.loads((bundle / "config.json").read_text())
    ckpt = Path(cfg["checkpoint"])
    ckpt = ckpt if ckpt.is_absolute() else bundle / ckpt
    voc = Path(vocabs) if vocabs else (bundle / cfg["vocabs"] if cfg.get("vocabs") else None)
    if voc is None:
        raise SystemExit(f"{bundle} keeps no vocabularies: pass --vocabs (python3 scripts/fetch_mac.py vocabs)")
    D, _ = pymal_run.official()
    calibrated = cfg.get("calibrator") is not None
    codes = json.loads((bundle / "group_codes.json").read_text()) if calibrated else {}
    server = Server.from_parts(
        cfg["args"],
        ckpt,
        D.get_vocabs(str(voc)),
        codes,
        bundle / cfg["calibrator"] if calibrated else None,
        threads,
        {"bundle": str(bundle), "source": cfg.get("source")},
    )
    if server.info["checkpoint_sha256"] != cfg["checkpoint_sha256"]:
        raise SystemExit(f"{ckpt} is not the checkpoint of {bundle} (SHA-256 differs)")
    return server


def predict(server: Server, request: dict, calibration: str = "iii") -> dict:
    """CVR per candidate. Without a calibrator in the bundle, cvr is the raw probability and calibration says "none"."""
    validate(request)
    server.oov.clear()
    p = server.score(server.request_tensors(request["user"], request["candidates"]))
    if server.gs is None:
        calibration, q = "none", p
    else:
        q = server.calibrate(p, request["candidates"]) if calibration == "iii" else server.glob.predict(p)
    return {
        "model": server.info["model"],
        "calibration": CALIBRATIONS[calibration],
        "candidates": [
            {
                "ad_feat0": c["ad_feat0"],
                "ad_feat1": c["ad_feat1"],
                "cvr": round(float(qi), 8),
                "cvr_uncalibrated": round(float(pi), 8),
            }
            for c, pi, qi in zip(request["candidates"], p, q)
        ],
        "values_not_in_vocabulary": dict(server.oov),
    }


def export(rid: str, out: Path, after: str = "train-17.parquet", calibrated: bool = True) -> dict:
    """Package a training run as a bundle. The 2 GB checkpoint stays in place and is referenced by path and SHA-256.

    A calibrator only fits the checkpoint whose predictions it was trained on (train-17). Exporting any other
    checkpoint requires calibrated=False."""
    from . import evaluation

    run = json.loads((evaluation.RUNS / rid / "run.json").read_text())
    ckpt = (evaluation.RUNS / rid / f"model_after_{Path(after).stem}.pt").resolve()
    sha = run["checkpoints"][after]["sha256"]
    if pymal_run.sha256(ckpt) != sha:
        raise SystemExit(f"{ckpt} differs from its run record")
    fitted = evaluation.prediction_file(
        "research", evaluation.CKPT, evaluation.checkpoint_sha(rid), evaluation.V1
    ).with_suffix(".json")
    fitted_sha = json.loads(fitted.read_text())["checkpoint_sha256"] if fitted.exists() else None
    if calibrated and fitted_sha != sha:
        raise SystemExit(
            f"the calibrator of {rid} was fitted on another checkpoint's predictions ({fitted}); "
            "export this checkpoint with --no-calibration to serve raw probabilities"
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    source = {"study": "E", "run": rid, "trained_through": after}
    if calibrated:
        shutil.copy(evaluation.calibrator_path(rid, "iii", "main"), out / "calibrator.npz")
        (out / "group_codes.json").write_text(json.dumps(evaluation.token_codes()))
        source |= {
            "calibrator_fitted_on": f"{evaluation.V1} predictions of checkpoint {fitted_sha[:12]}",
            "lambda_chosen_on": evaluation.V2,
        }
    cfg = {
        "args": run["args"],
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha,
        "calibrator": "calibrator.npz" if calibrated else None,
        "calibration": "iii" if calibrated else "none",
        "vocabs": None,
        "source": source,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    return cfg


def request_from(parquet: Path, row: int, k: int) -> dict:
    """Build an example request from real MAC rows: the user of `row`, with up to k of that user's clicks as candidates.

    Not a logged request. The clicks may come from different ad contexts."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    user_cols, cand_cols = columns()
    users = pq.read_table(parquet, columns=["user_id"]).column("user_id")
    uid = users[row].as_py()
    idx = np.flatnonzero(pc.equal(users, uid).to_numpy(zero_copy_only=False))
    idx = np.array([row, *(i for i in idx if i != row)][:k], dtype=np.int64)  # int even when the user has one click
    t = pq.read_table(parquet, columns=["user_id", *user_cols, *cand_cols]).take(idx).to_pylist()
    return {
        "user": {c: t[0][c] for c in ["user_id", *user_cols]},
        "candidates": [{c: r[c] for c in cand_cols} for r in t],
    }


def main(argv: list[str]) -> None:
    if argv[:1] == ["export"]:
        ap = argparse.ArgumentParser(prog="cvr.predictor export")
        ap.add_argument("run")
        ap.add_argument("out", type=Path)
        ap.add_argument("--after", default="train-17.parquet")
        ap.add_argument("--no-calibration", action="store_true", help="a checkpoint without its own calibrator")
        a = ap.parse_args(argv[1:])
        print(json.dumps(export(a.run, a.out, a.after, calibrated=not a.no_calibration), indent=2))
        return
    if argv[:1] == ["request-from"]:
        ap = argparse.ArgumentParser(prog="cvr.predictor request-from")
        ap.add_argument("parquet", type=Path)
        ap.add_argument("row", type=int)
        ap.add_argument("--candidates", type=int, default=3, help="at most this many of the user's clicks (default 3)")
        a = ap.parse_args(argv[1:])
        print(json.dumps(request_from(a.parquet, a.row, a.candidates)))
        return
    ap = argparse.ArgumentParser(
        prog="cvr.predictor", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("request", type=Path, help="request JSON, or - for standard input")
    ap.add_argument("--bundle", type=Path, default=EXAMPLE)
    ap.add_argument("--vocabs", type=Path, default=None)
    ap.add_argument("--calibration", choices=["iii", "ii"], default="iii")
    ap.add_argument("--jsonl", action="store_true", help="one request per input line, one response per output line")
    a = ap.parse_args(argv)
    server = load(a.bundle, a.vocabs)
    if not a.jsonl:
        text = sys.stdin.read() if str(a.request) == "-" else a.request.read_text()
        try:
            print(json.dumps(predict(server, json.loads(text), a.calibration), indent=2))
        except (RequestError, json.JSONDecodeError) as e:
            raise SystemExit(f"invalid request: {e}") from None
        return
    stream = sys.stdin if str(a.request) == "-" else a.request.open()
    for line in stream:
        if not line.strip():
            continue
        try:
            out = predict(server, json.loads(line), a.calibration)
        except Exception as e:
            out = {"error": f"{type(e).__name__}: {e}"}
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
