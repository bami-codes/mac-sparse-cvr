"""Calibrate the official models' predictions on V1 (train-18) and choose the shrinkage strength on V2 (train-19).

    uv run python -m cvr.evaluation calibrate runs/eval/config_choice.json

The inputs are the train-17 checkpoints' predictions on V1 and V2, found through each checkpoint's SHA-256 in its
run record. Every prediction file is checked row by row against labels read again from the source parquet, so a
row-order slip cannot reach the calibration.

Calibrations: (ii) global Platt, (iii) group-shrunk Platt solved exactly per group, (iii-0) group intercept only.
Lambda is picked per model and method by the seed mean on V2: G1 macro LogLoss (main rule) or G1 group-total Poisson
deviance (aggregate rule). The V2 numbers are selection data, recorded so the dry run can reproduce them.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from scipy import sparse

from . import groups, metrics
from .calibration import (
    GlobalPlatt,
    fit_group_intercept_exact,
    fit_group_shrunk_exact,
    load_group_calibrators,
    save_group_calibrators,
)
from .metrics import poisson_deviance

RUNS = Path("runs/train")
TEST_PATH = Path("data/raw/mac/test/test.parquet")
LAMBDAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
SEEDS = [2026, 2027, 2028]
V1, V2, TEST = "train-18.parquet", "train-19.parquet", "test.parquet"
CKPT = "model_after_train-17"
BOOT = {"reps": 1000, "seed": 20261119}
METHODS = ("iii", "iii0")
WEIGHTED = "MoAE-g1w2"  # Q4: the chosen MoAE retrained with the G1 clicks' main loss weighted 2x


def source(name: str) -> Path:
    return TEST_PATH if name == TEST else groups.DATA / name


def token_codes() -> dict[str, int]:
    tokens = pl.read_parquet(groups.OUT / "train_stats_ad_feat1.parquet")["ad_feat1"].to_list()
    return {t: i for i, t in enumerate(sorted(tokens))}


def file_columns(name: str, codes: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    t = pl.from_arrow(pq.read_table(source(name), columns=["ad_feat1", "last"]))
    g = t["ad_feat1"].replace_strict(codes, default=-1, return_dtype=pl.Int64).to_numpy()
    return g, (t["last"].to_numpy() > 0).astype(np.int8)


def group_sets(codes: dict[str, int]) -> dict[str, np.ndarray]:
    lists = json.loads((groups.OUT / "group_lists.json").read_text())
    return {k: np.array(sorted(codes[t] for t in lists[k])) for k in ("G1", "C")}


def run_id(model: str, lr: float, lamda: float, seed: int) -> str:
    return f"{model}_lr{lr:g}_lam{lamda:g}_s{seed}"


def checkpoint_sha(rid: str, after: str = "train-17.parquet") -> str:
    return json.loads((RUNS / rid / "run.json").read_text())["checkpoints"][after]["sha256"]


def prediction_file(question_dir: str, stem: str, sha: str, name: str) -> Path:
    """cvr.pymal_run predict writes <out>/<checkpoint stem>_<first 12 of its SHA-256>/<file>.npz: predictions are
    found through the checkpoint itself, so all checkpoints of one claim of the test can share one output folder."""
    return RUNS / question_dir / f"{stem}_{sha[:12]}" / f"{Path(name).stem}.npz"


def predictions(rid: str, name: str, y_file: np.ndarray) -> dict:
    sha = checkpoint_sha(rid)
    f = prediction_file("research", CKPT, sha, name)
    if not f.exists():
        raise FileNotFoundError(f"{rid} {name}: {f}")
    hits = [f]
    z = dict(np.load(hits[0]))
    meta = json.loads(hits[0].with_suffix(".json").read_text())
    if meta["checkpoint_sha256"] != sha:
        raise ValueError(f"{f}: made with checkpoint {meta['checkpoint_sha256'][:12]}, the run's is {sha[:12]}")
    if z["p_last"].size != y_file.size or not np.array_equal(z["y_last"].astype(np.int8), y_file):
        raise ValueError(f"{hits[0]}: labels differ from {name} read again in file order")
    if hashlib.sha256(hits[0].read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError(f"{hits[0]} differs from its record")
    return {
        "p": z["p_last"].astype(np.float64),
        "user": z["user"],
        "file": str(hits[0]),
        "sha256": meta["sha256"],
        "checkpoint_sha256": meta["checkpoint_sha256"],
    }


# ------------------------------------------------------------------ metrics (the same function for V2, dry run and test)
def group_metrics(y: np.ndarray, q: np.ndarray, g: np.ndarray, members: np.ndarray) -> dict:
    m = np.isin(g, members)
    r = metrics.group_report(y[m], q[m], g[m])
    pg = metrics.per_group(y[m], q[m], g[m])
    return {
        "macro_logloss": r["macro_logloss"],
        "click_logloss": r["click_weighted_logloss"],
        "poisson_mean": float(poisson_deviance(pg["O"].to_numpy(), pg["E"].to_numpy()).mean()),
        "within_auc_macro": r["within_group_auc_macro"],
        "O_over_E": r["pooled_O_over_E"],
        "groups": r["groups"],
        "rows": r["rows"],
    }


def evaluate(y: np.ndarray, q: np.ndarray, g: np.ndarray, sets: dict[str, np.ndarray], users=None) -> dict:
    """Group metrics on every set, and overall LogLoss, AUC, O/E and (with users) the click-weighted GAUC."""
    from sklearn.metrics import roc_auc_score

    overall = {
        "logloss": float(metrics.bce(y, q).mean()),
        "auc": float(roc_auc_score(y, q)),
        "O_over_E": float(y.sum() / q.sum()),
    }
    if users is not None:  # GAUC needs each row's user
        overall["gauc"] = metrics.gauc(y, q, users)["gauc"]
    return {k: group_metrics(y, q, g, v) for k, v in sets.items()} | {"overall": overall}


# Planned values the V2 selection run did not compute. The dry run records their V2 values instead of reproducing
# them, and refuses any other difference between the two sets of keys.
ADDED_AFTER_SELECTION = {"Q3 click ii: (MoAE - BASE) on G1 minus on C", "Q3 click raw: (MoAE - BASE) on G1 minus on C"}


def did_click_bootstrap(y, a: list, b: list, g, users, set_1, set_2, reps: int, seed: int) -> dict:
    """(click LogLoss_1(A) - click LogLoss_1(B)) - (the same on set 2): each set weights its clicks equally; one user
    resample per replicate for both sets; seed pairs averaged."""
    _, ui = np.unique(users, return_inverse=True)
    U = ui.max() + 1
    d = np.mean([metrics.bce(y, pa) - metrics.bce(y, pb) for pa, pb in zip(a, b)], axis=0)
    parts = []
    for members in (set_1, set_2):
        m = np.isin(g, members)
        parts.append((np.bincount(ui[m], minlength=U).astype(float), np.bincount(ui[m], weights=d[m], minlength=U)))

    def stat(w):
        v = [float(w @ tot / (w @ n)) for n, tot in parts]
        return v[0] - v[1], v

    point, (d1, d2) = stat(np.ones(U))
    rng = np.random.default_rng(seed)
    reps_ = np.array([stat(np.bincount(rng.integers(U, size=U), minlength=U).astype(float))[0] for _ in range(reps)])
    lo, hi = np.quantile(reps_, [0.025, 0.975])
    return {
        "did": point,
        "diff_set_1": d1,
        "diff_set_2": d2,
        "ci95": [float(lo), float(hi)],
        "reps": reps,
        "users": int(U),
        "seed_pairs": len(a),
    }


def did_bootstrap(y, a: list, b: list, g, users, set_1, set_2, reps: int, seed: int) -> dict:
    """(macro_1(A) - macro_1(B)) - (macro_2(A) - macro_2(B)) with one user resample per replicate for both sets
    (the same scheme as cvr.metrics.paired_bootstrap_macro_diff); seed pairs averaged as in that function."""
    _, ui = np.unique(users, return_inverse=True)
    U = ui.max() + 1
    d = np.mean([metrics.bce(y, pa) - metrics.bce(y, pb) for pa, pb in zip(a, b)], axis=0)
    parts = []
    for members in (set_1, set_2):
        m = np.isin(g, members)
        _, gi = np.unique(g[m], return_inverse=True)
        G = gi.max() + 1
        parts.append(
            (
                sparse.csr_matrix((np.ones(m.sum()), (ui[m], gi)), shape=(U, G)),
                sparse.csr_matrix((d[m], (ui[m], gi)), shape=(U, G)),
            )
        )

    def stat(w):
        out = []
        for n, s in parts:
            cnt, tot = n.T @ w, s.T @ w
            keep = cnt > 0
            out.append(float((tot[keep] / cnt[keep]).mean()))
        return out[0] - out[1], out

    point, (d1, d2) = stat(np.ones(U))
    rng = np.random.default_rng(seed)
    reps_ = np.array([stat(np.bincount(rng.integers(U, size=U), minlength=U).astype(float))[0] for _ in range(reps)])
    lo, hi = np.quantile(reps_, [0.025, 0.975])
    return {
        "did": point,
        "diff_set_1": d1,
        "diff_set_2": d2,
        "ci95": [float(lo), float(hi)],
        "reps": reps,
        "users": int(U),
        "seed_pairs": len(a),
    }


def click_diff_bootstrap(y, a: list, b: list, users, reps: int, seed: int) -> dict:
    """Click-weighted LogLoss(A) - LogLoss(B) over all rows, one user resample per replicate, seed pairs averaged."""
    _, ui = np.unique(users, return_inverse=True)
    U = ui.max() + 1
    d = np.mean([metrics.bce(y, pa) - metrics.bce(y, pb) for pa, pb in zip(a, b)], axis=0)
    n, tot = np.bincount(ui, minlength=U).astype(float), np.bincount(ui, weights=d, minlength=U)
    rng = np.random.default_rng(seed)
    reps_ = []
    for _ in range(reps):
        w = np.bincount(rng.integers(U, size=U), minlength=U).astype(float)
        reps_.append(float(w @ tot / (w @ n)))
    lo, hi = np.quantile(reps_, [0.025, 0.975])
    return {
        "diff": float(tot.sum() / n.sum()),
        "ci95": [float(lo), float(hi)],
        "reps": reps,
        "users": int(U),
        "seed_pairs": len(a),
        "per_pair_diff": [float((metrics.bce(y, pa) - metrics.bce(y, pb)).mean()) for pa, pb in zip(a, b)],
    }


def q4_comparisons(y, g, users, sets, get, iii_w: str, iii_u: str) -> dict:
    """Q4 comparisons, the same code on V2 and on the test. `get(model, name)` returns the predictions in seed order."""
    m1 = np.isin(g, sets["G1"])

    def macro(a, b, label):
        return metrics.paired_bootstrap_macro_diff(
            y[m1], [x[m1] for x in a], [x[m1] for x in b], g[m1], users[m1], **BOOT
        ) | {"compared": label}

    w3, u3 = get(WEIGHTED, iii_w), get("MoAE", iii_u)
    pois = [
        group_metrics(y, a, g, sets["G1"])["poisson_mean"] - group_metrics(y, b, g, sets["G1"])["poisson_mean"]
        for a, b in zip(w3, u3)
    ]
    return {
        "Q4: weighted iii - plain iii": macro(w3, u3, f"{WEIGHTED} ({iii_w}) - MoAE ({iii_u}), G1 macro"),
        "Q4: weighted ii - plain ii": macro(
            get(WEIGHTED, "ii"), get("MoAE", "ii"), f"{WEIGHTED} (ii) - MoAE (ii), G1 macro"
        ),
        "Q4 guard: weighted iii - plain iii, all clicks": click_diff_bootstrap(y, w3, u3, users, **BOOT),
        "Q4 guard: weighted iii - plain iii, G1 Poisson deviance": {
            "diff": float(np.mean(pois)),
            "per_pair_diff": pois,
        },
    }


# ------------------------------------------------------------------ calibration on V1, choice on V2
def fit_run(p1, y1, g1) -> tuple[GlobalPlatt, dict, dict]:
    glob = GlobalPlatt.fit(p1, y1)
    fits, info = {}, {}
    for lam in LAMBDAS:
        for method, fit in (("iii", fit_group_shrunk_exact), ("iii0", fit_group_intercept_exact)):
            gs, si = fit(p1, y1, g1, glob, lam)
            if not gs.converged or gs.n_nonpositive_slopes:
                raise RuntimeError(f"{method} lambda {lam}: converged={gs.converged}, {si}")
            fits[(method, lam)], info[f"{method}|lam{lam:g}"] = gs, si
    return glob, fits, info


def apply(glob, fits, p, g) -> dict[str, np.ndarray]:
    out = {"raw": p, "ii": glob.predict(p)}
    for (method, lam), gs in fits.items():
        out[f"{method}_lam{lam:g}"] = gs.predict(p, g)
    return out


def calibrate(choice_path: Path) -> dict:
    choice = json.loads(Path(choice_path).read_text())  # {"MoAE": {"lr":..,"lamda":..}, "BASE": {...}}
    codes = token_codes()
    sets = group_sets(codes)
    g1, y1 = file_columns(V1, codes)
    g2, y2 = file_columns(V2, codes)
    out_dir = groups.OUT / "q2"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, solver, q2, used, users2 = {}, {}, {}, {}, None
    cal_dir = out_dir / "calibrators"
    cal_dir.mkdir(exist_ok=True)
    fitted = {}
    for model, c in choice.items():
        for s in SEEDS:
            rid = run_id(model, c["lr"], c["lamda"], s)
            a, b = predictions(rid, V1, y1), predictions(rid, V2, y2)
            if users2 is None:
                users2 = b["user"]
            elif not np.array_equal(users2, b["user"]):
                raise ValueError("V2 users differ between runs")
            glob, fits, info = fit_run(a["p"], y1, g1)
            fitted[rid] = (glob, fits)
            solver[rid] = info
            q2[rid] = apply(glob, fits, b["p"], g2)
            used[rid] = {"v1": a, "v2": b} | {"checkpoint_sha256": b["checkpoint_sha256"]}
            for name, q in q2[rid].items():
                rows[f"{rid}|{name}"] = evaluate(y2, q, g2, sets)
            print(rid, {k: round(rows[f"{rid}|{k}"]["G1"]["macro_logloss"], 6) for k in ("raw", "ii")}, flush=True)

    def seed_mean(model, name, path=("G1", "macro_logloss")):
        c = choice[model]
        return float(
            np.mean([rows[f"{run_id(model, c['lr'], c['lamda'], s)}|{name}"][path[0]][path[1]] for s in SEEDS])
        )

    lam = {
        m: {
            meth: {
                "main": min(LAMBDAS, key=lambda x: seed_mean(m, f"{meth}_lam{x:g}")),
                "aggregate": min(LAMBDAS, key=lambda x: seed_mean(m, f"{meth}_lam{x:g}", ("G1", "poisson_mean"))),
            }
            for meth in METHODS
        }
        for m in choice
    }
    for rid, (glob, fits) in fitted.items():  # the chosen calibrators, saved for the test
        model = rid.split("_")[0]
        for meth in METHODS:
            for rule in ("main", "aggregate"):
                save_group_calibrators(glob, fits[(meth, lam[model][meth][rule])], calibrator_path(rid, meth, rule))

    def seeds_of(model, name):
        c = choice[model]
        return [q2[run_id(model, c["lr"], c["lamda"], s)][name] for s in SEEDS]

    m1 = np.isin(g2, sets["G1"])

    def boot(model, a_name, b_name):
        r = metrics.paired_bootstrap_macro_diff(
            y2[m1],
            [x[m1] for x in seeds_of(model, a_name)],
            [x[m1] for x in seeds_of(model, b_name)],
            g2[m1],
            users2[m1],
            **BOOT,
        )
        return r | {"compared": f"{model}: {a_name} - {b_name}"}

    v2 = {}
    for model in choice:
        iii, iii0 = f"iii_lam{lam[model]['iii']['main']:g}", f"iii0_lam{lam[model]['iii0']['main']:g}"
        agg = f"iii_lam{lam[model]['iii']['aggregate']:g}"
        v2[f"{model}: iii - ii"] = boot(model, iii, "ii")
        v2[f"{model}: iii0 - ii"] = boot(model, iii0, "ii")
        v2[f"{model}: iii0 - iii"] = boot(model, iii0, iii)
        v2[f"{model}: iii (aggregate rule) - ii"] = boot(model, agg, "ii")
    if {"MoAE", "BASE"} <= set(choice):
        for name in ("ii", "raw"):
            v2[f"Q3 {name}: (MoAE - BASE) on G1 minus on C"] = did_bootstrap(
                y2, seeds_of("MoAE", name), seeds_of("BASE", name), g2, users2, sets["G1"], sets["C"], **BOOT
            )
    if {"MoAE", WEIGHTED} <= set(choice):
        v2 |= q4_comparisons(
            y2,
            g2,
            users2,
            sets,
            seeds_of,
            f"iii_lam{lam[WEIGHTED]['iii']['main']:g}",
            f"iii_lam{lam['MoAE']['iii']['main']:g}",
        )
    rec = {
        "choice": choice,
        "lambdas": LAMBDAS,
        "seeds": SEEDS,
        "lambda_chosen": lam,
        "v2_rows": rows,
        "v2_bootstrap": v2,
        "solver": solver,
        "inputs": {
            rid: {"v1": u["v1"]["sha256"], "v2": u["v2"]["sha256"], "checkpoint_sha256": u["checkpoint_sha256"]}
            for rid, u in used.items()
        },
        "groups_sha256": json.loads((groups.OUT / "groups.json").read_text())["group_lists_sha256"],
        "note": "V2 = selection data, not a result",
    }
    (out_dir / "selection.json").write_text(json.dumps(rec, indent=2, default=str))
    return rec


def calibrator_path(rid: str, meth: str, rule: str) -> Path:
    """Spelled with .npz: run ids contain dots (lr0.003), and with_suffix would otherwise cut the name at the last
    one, making every run write the same file."""
    return groups.OUT / "q2" / "calibrators" / f"{rid}__{meth}_{rule}.npz"


def load_calibrated(rid: str, meth: str, rule: str, p: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    glob, gs = load_group_calibrators(calibrator_path(rid, meth, rule))
    return glob.predict(p), gs.predict(p, g)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "calibrate":
        r = calibrate(Path(sys.argv[2]))
        print(
            json.dumps(
                {
                    "lambda_chosen": r["lambda_chosen"],
                    "v2_bootstrap": {
                        k: {kk: v.get(kk) for kk in ("diff", "did", "ci95", "per_pair_diff")}
                        for k, v in r["v2_bootstrap"].items()
                    },
                },
                indent=2,
            )
        )
    else:
        raise SystemExit(__doc__)
