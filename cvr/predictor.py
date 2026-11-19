"""CVR predictor: one user's request in, a calibrated conversion probability per ad candidate out.

    uv run python -m cvr.predictor examples/request.json                    # the example model (synthetic data)
    uv run python -m cvr.predictor request.json --bundle bundles/BASE_train17 --vocabs data/raw/mac/vocabs
    uv run python -m cvr.predictor - --jsonl < requests.jsonl               # load once, one request per line
    uv run python -m cvr.predictor export BASE_lr0.003_lam0.3_s2026 bundles/BASE_train17   # a study E model
    uv run python -m cvr.predictor request-from data/raw/mac/train/train-19.parquet 123 --candidates 3 > request.json

A model is a folder (a bundle):
- config.json: the official model's training arguments, the checkpoint (a path inside the folder or elsewhere) and its
  SHA-256, the calibrator file, the vocabulary folder (inside the folder, or given with --vocabs), where it came from;
- calibrator.npz: global Platt (ii) and group-shrunk Platt (iii) on the ad group `ad_feat1`;
- group_codes.json: ad group token -> calibrator group (a token absent from it uses the pooled "unseen" group).

A request is {"user": {...}, "candidates": [{...}, ...]}, with MAC's raw columns: the user's features (`user_feat0`-`5`)
and purchase history (`ad_feat0_seq`-`ad_feat2_seq`, one list each, equal lengths), and for each candidate its ad and
context features (`ad_feat0`-`5`, `context_feat0`-`3`) and its similarity to each history item (`mm_feat0_seq`,
`mm_feat1_seq`, the history's length). The user's history is encoded once for all candidates.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from . import pymal_run as R
from .serving_e import ServerE

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "model"
CALIBRATIONS = {"iii": "group-shrunk Platt on the ad group (ad_feat1)", "ii": "global Platt"}


class RequestError(ValueError):
    pass


def columns() -> tuple[list[str], list[str]]:
    D, _ = R.official()
    return D.USER_COLS[1:] + D.SEQ_COLS, D.AD_COLS + D.MM_COLS


def validate(request: dict) -> None:
    D, _ = R.official()
    user_cols, cand_cols = columns()
    if not isinstance(request, dict) or not isinstance(request.get("user"), dict):
        raise RequestError('a request is {"user": {...}, "candidates": [{...}, ...]}')
    cands = request.get("candidates")
    if not isinstance(cands, list) or not cands:
        raise RequestError("candidates must be a non-empty list")
    user = request["user"]
    if missing := [c for c in user_cols if c not in user]:
        raise RequestError(f"user lacks {missing}")
    lengths = {c: len(user[c]) for c in D.SEQ_COLS if isinstance(user[c], list)}
    if len(lengths) != len(D.SEQ_COLS) or len(set(lengths.values())) != 1:
        raise RequestError(f"the purchase history columns must be lists of one length: {lengths}")
    n = lengths[D.SEQ_COLS[0]]
    for i, c in enumerate(cands):
        if missing := [k for k in cand_cols if k not in c]:
            raise RequestError(f"candidate {i} lacks {missing}")
        bad = [k for k in D.MM_COLS if not isinstance(c[k], list) or len(c[k]) != n]
        if bad:
            raise RequestError(f"candidate {i}: {bad} must hold one similarity per history item ({n})")


def load(bundle: Path = EXAMPLE, vocabs: Path | None = None, threads: int = 1) -> ServerE:
    bundle = Path(bundle)
    cfg = json.loads((bundle / "config.json").read_text())
    ckpt = Path(cfg["checkpoint"])
    ckpt = ckpt if ckpt.is_absolute() else bundle / ckpt
    voc = Path(vocabs) if vocabs else (bundle / cfg["vocabs"] if cfg.get("vocabs") else None)
    if voc is None:
        raise SystemExit(f"{bundle} keeps no vocabularies: pass --vocabs (python3 scripts/fetch_mac.py vocabs)")
    D, _ = R.official()
    codes = json.loads((bundle / "group_codes.json").read_text())
    server = ServerE.from_parts(cfg["args"], ckpt, D.get_vocabs(str(voc)), codes, bundle / cfg["calibrator"], threads,
                                {"bundle": str(bundle), "source": cfg.get("source")})
    if server.info["checkpoint_sha256"] != cfg["checkpoint_sha256"]:
        raise SystemExit(f"{ckpt} is not the checkpoint of {bundle} (SHA-256 differs)")
    return server


def predict(server: ServerE, request: dict, calibration: str = "iii") -> dict:
    validate(request)
    server.oov.clear()
    p = server.score(server.request_tensors(request["user"], request["candidates"]))
    q = server.calibrate(p, request["candidates"]) if calibration == "iii" else server.glob.predict(p)
    return {"model": server.info["model"], "calibration": CALIBRATIONS[calibration],
            "candidates": [{"ad_feat0": c["ad_feat0"], "ad_feat1": c["ad_feat1"], "cvr": round(float(qi), 8),
                            "cvr_uncalibrated": round(float(pi), 8)} for c, pi, qi in zip(request["candidates"], p, q)],
            "values_not_in_vocabulary": dict(server.oov)}


def export(rid: str, out: Path, after: str = "train-17.parquet", calibration: str = "iii") -> dict:
    """A study E model as a bundle: small files are copied, the checkpoint is referenced by path and SHA-256."""
    from . import study_e_eval as Q
    run = json.loads((Q.RUNS / rid / "run.json").read_text())
    ckpt = (Q.RUNS / rid / f"model_after_{Path(after).stem}.pt").resolve()
    if R.sha256(ckpt) != run["checkpoints"][after]["sha256"]:
        raise SystemExit(f"{ckpt} differs from its run record")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy(Q.calibrator_path(rid, calibration, "main"), out / "calibrator.npz")
    (out / "group_codes.json").write_text(json.dumps(Q.token_codes()))
    cfg = {"args": run["args"], "checkpoint": str(ckpt), "checkpoint_sha256": run["checkpoints"][after]["sha256"],
           "calibrator": "calibrator.npz", "calibration": calibration, "vocabs": None,
           "source": {"study": "E", "run": rid, "trained_through": after,
                      "calibrator_fitted_on": "train-18.parquet", "lambda_chosen_on": "train-19.parquet"}}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    return cfg


def request_from(parquet: Path, row: int, k: int) -> dict:
    """A real request from MAC rows: the user of `row` and up to k rows of that user in the same file (each row's
    similarity sequences belong to that user's history, so the candidates are consistent with the user)."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    user_cols, cand_cols = columns()
    users = pq.read_table(parquet, columns=["user_id"]).column("user_id")
    uid = users[row].as_py()
    idx = np.flatnonzero(pc.equal(users, uid).to_numpy(zero_copy_only=False))
    idx = np.r_[row, [i for i in idx if i != row]][:k]
    t = pq.read_table(parquet, columns=["user_id", *user_cols, *cand_cols]).take(idx).to_pylist()
    return {"user": {c: t[0][c] for c in ["user_id", *user_cols]}, "candidates": [{c: r[c] for c in cand_cols} for r in t]}


def main(argv: list[str]) -> None:
    if argv[:1] == ["export"]:
        ap = argparse.ArgumentParser(prog="cvr.predictor export")
        ap.add_argument("run")
        ap.add_argument("out", type=Path)
        ap.add_argument("--after", default="train-17.parquet")
        a = ap.parse_args(argv[1:])
        print(json.dumps(export(a.run, a.out, a.after), indent=2))
        return
    if argv[:1] == ["request-from"]:
        ap = argparse.ArgumentParser(prog="cvr.predictor request-from")
        ap.add_argument("parquet", type=Path)
        ap.add_argument("row", type=int)
        ap.add_argument("--candidates", type=int, default=3)
        a = ap.parse_args(argv[1:])
        print(json.dumps(request_from(a.parquet, a.row, a.candidates)))
        return
    ap = argparse.ArgumentParser(prog="cvr.predictor", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("request", type=Path, help="request JSON, or - for standard input")
    ap.add_argument("--bundle", type=Path, default=EXAMPLE)
    ap.add_argument("--vocabs", type=Path, default=None)
    ap.add_argument("--calibration", choices=sorted(CALIBRATIONS), default="iii")
    ap.add_argument("--jsonl", action="store_true", help="one request per input line, one response per output line")
    a = ap.parse_args(argv)
    text = sys.stdin.read() if str(a.request) == "-" else a.request.read_text()
    server = load(a.bundle, a.vocabs)
    if not a.jsonl:
        try:
            print(json.dumps(predict(server, json.loads(text), a.calibration), indent=2))
        except RequestError as e:
            raise SystemExit(f"invalid request: {e}") from None
        return
    for line in filter(str.strip, text.splitlines()):  # a bad request answers with its error and the next one goes on
        try:
            print(json.dumps(predict(server, json.loads(line), a.calibration)), flush=True)
        except (RequestError, json.JSONDecodeError) as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
