"""Mechanism study A (docs/DESIGN.md §15.1): with the fitted M2 models held fixed, how does the gap between
global (ii) and group-shrunk (iii) calibration change when only the amount of calibration data changes?

Development data only: V1 train-18 (calibration fit) and V2 train-19 (evaluation) of the first study's
development run. V1 users are subsampled; every subsample is shared by all seeds and calibrators.

    PYTHONPATH=. uv run python scripts/calibration_data_study.py runs/dev_train15_17_m3/20261117T110713Z runs/calib_study \
        --md docs/results/calibration_data_study.md
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from cvr import groups as G
from cvr.artifacts import load_gbdt
from cvr.calibration import GlobalPlatt, GroupShrunkPlatt
from cvr.features import featurize, read_raw
from cvr.final_eval import verify_freeze, vocab_frames
from cvr.metrics import bce
from cvr.schema import GROUP_KEY

FRACTIONS = [1.0, 0.5, 0.25, 0.1, 0.05]
REPS = 5  # subsample draws for every fraction below 1
LAMBDAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
SUB_SEED = 20261118
# V2 G1 groups by the conversions of the group in the V1 rows actually used for calibration
BINS = ["absent from V1", "0", "1-2", "3-9", "10+"]
REPRO_TOL = 1e-9


def prepare(run: Path, out: Path) -> dict:
    """V1 / V2 predictions of every M2 seed, cached (computed once with the run's own vocab and group stats)."""
    cache = {s: out / f"preds_{s}.parquet" for s in ("v1", "v2")}
    meta_path = out / "prepare.json"
    if all(p.exists() for p in cache.values()) and meta_path.exists():
        return json.loads(meta_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    manifest = verify_freeze(run)
    cfg, seeds = manifest["config"], manifest["seeds"]
    vocab = vocab_frames(run / "vocab")
    stats = pl.read_parquet(run / "frozen_group_stats.parquet")
    gc = manifest["groups"]["definition"]
    models = {s: load_gbdt(run / "models" / f"M2_s{s}") for s in seeds}
    root = Path(cfg["data"]["root"])
    rows = {}
    for split in ("v1", "v2"):
        df = pl.concat([featurize(read_raw(root / f), vocab) for f in cfg["data"][split]])
        df = pl.concat([df, G.classify(df, stats, gc["n_min"], gc["c_min"], gc["c_dense"])], how="horizontal")
        cols = {"user_key": df["user_key"], "y": df["y"], "g": df[GROUP_KEY], "group_class": df["group_class"]}
        for s, m in models.items():
            cols[f"p_s{s}"] = m.predict(df)
        pl.DataFrame(cols).write_parquet(cache[split])
        rows[split] = df.height
    meta = {"run": str(run), "seeds": seeds, "files": {s: cfg["data"][s] for s in ("v1", "v2")}, "rows": rows}
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


# ------------------------------------------------------------------ worker state (loaded once per process)
_S: dict = {}


def _init(out: str, seeds: list[int]) -> None:
    v1, v2 = pl.read_parquet(Path(out) / "preds_v1.parquet"), pl.read_parquet(Path(out) / "preds_v2.parquet")
    g1 = v2["group_class"].to_numpy() == "G1"
    _S.update(
        seeds=seeds,
        u1=v1["user_key"].to_numpy(),
        y1=v1["y"].to_numpy().astype(np.float64),
        g1=v1["g"].to_numpy(),
        p1={s: v1[f"p_s{s}"].to_numpy() for s in seeds},
        y2=v2["y"].to_numpy()[g1].astype(np.float64),
        g2=v2["g"].to_numpy()[g1],
        p2={s: v2[f"p_s{s}"].to_numpy() for s in seeds},
        g2_all=v2["g"].to_numpy(),
        m2=g1,
        users=np.unique(v1["user_key"].to_numpy()),
    )


def subsample(frac: float, rep: int) -> np.ndarray:
    if frac >= 1.0:
        return np.ones(_S["u1"].size, bool)
    rng = np.random.default_rng([SUB_SEED, rep, round(frac * 1000)])
    pick = rng.choice(_S["users"], size=round(frac * _S["users"].size), replace=False)
    return np.isin(_S["u1"], pick)


def group_table(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> pl.DataFrame:
    return (
        pl.DataFrame({"g": g, "y": y, "p": p, "l": bce(y, p)})
        .group_by("g")
        .agg(pl.col("y").sum().alias("O"), pl.col("p").sum().alias("E"), pl.col("l").mean().alias("ll"))
        .sort("g")
    )


def summarize_groups(t: pl.DataFrame) -> dict:
    o, e = t["O"].to_numpy(), t["E"].to_numpy()
    dev = 2 * (np.where(o > 0, o * np.log(np.where(o > 0, o, 1) / e), 0.0) - (o - e))
    pos = o > 0
    return {
        "macro_logloss": float(t["ll"].mean()),
        "poisson_deviance": float(dev.mean()),
        "old_median_abs_ratio_dev": float(np.median(np.abs(o[pos] / e[pos] - 1))),
    }


def task(args: tuple) -> dict:
    seed, frac, rep = args
    t0 = time.time()
    m = subsample(frac, rep)
    y1, g1, p1 = _S["y1"][m], _S["g1"][m], _S["p1"][seed][m]
    p2_all = _S["p2"][seed]
    # calibration evidence per V2 G1 group, from the V1 rows actually used
    ev = (
        pl.DataFrame({"g": g1, "y": y1})
        .group_by("g")
        .agg(pl.len().alias("v1_clicks"), pl.col("y").sum().alias("v1_conv"))
    )
    glob = GlobalPlatt.fit(p1, y1)
    t_ii = group_table(_S["y2"], glob.predict(p2_all)[_S["m2"]], _S["g2"])
    bins = (
        t_ii.select("g")
        .join(ev, on="g", how="left", maintain_order="left")
        .with_columns(
            pl.when(pl.col("v1_clicks").is_null())
            .then(pl.lit("absent from V1"))
            .when(pl.col("v1_conv") == 0)
            .then(pl.lit("0"))
            .when(pl.col("v1_conv") <= 2)
            .then(pl.lit("1-2"))
            .when(pl.col("v1_conv") <= 9)
            .then(pl.lit("3-9"))
            .otherwise(pl.lit("10+"))
            .alias("bin")
        )["bin"]
        .to_numpy()
    )
    res = {
        "seed": seed,
        "frac": frac,
        "rep": rep,
        "v1_rows": int(m.sum()),
        "v1_conversions": int(y1.sum()),
        "g1_groups": int(t_ii.height),
        "groups_per_bin": {b: int((bins == b).sum()) for b in BINS},
        "ii": summarize_groups(t_ii),
        "iii": {},
    }
    for lam in LAMBDAS:
        gs = GroupShrunkPlatt.fit(p1, y1, g1, glob, lam)
        t = group_table(_S["y2"], gs.predict(p2_all, _S["g2_all"])[_S["m2"]], _S["g2"])
        d = t["ll"].to_numpy() - t_ii["ll"].to_numpy()
        res["iii"][f"{lam:g}"] = summarize_groups(t) | {
            "converged": gs.converged,
            "nonpositive_slopes": gs.n_nonpositive_slopes,
            # sum over the groups of a bin / all G1 groups = that bin's share of the macro LogLoss difference
            "macro_diff_by_bin": {b: float(d[bins == b].sum() / d.size) for b in BINS},
            "mean_group_diff_by_bin": {b: (float(d[bins == b].mean()) if (bins == b).any() else None) for b in BINS},
            "max_abs_diff_absent": float(np.abs(d[bins == "absent from V1"]).max())
            if (bins == "absent from V1").any()
            else 0.0,
        }
    res["seconds"] = round(time.time() - t0, 1)
    return res


def reproduce(run: Path, seeds: list[int]) -> dict:
    """Full-V1 (ii) and (iii) lam=100 must reproduce the development run's stored V2 numbers."""
    stored = json.loads((run / "results.json").read_text())["v2_rows"]
    out = {}
    for s in seeds:
        r = task((s, 1.0, 0))
        for key, mine in (
            (f"M2_s{s}|ii", r["ii"]["macro_logloss"]),
            (f"M2_s{s}|iii_lam100", r["iii"]["100"]["macro_logloss"]),
        ):
            ref = stored[key]["G1"]["macro_logloss"]
            if abs(mine - ref) > REPRO_TOL:
                raise SystemExit(f"reproduction failed for {key}: {mine} vs stored {ref}")
            out[key] = {"recomputed": mine, "stored": ref}
    return out


def write_md(path: Path, meta: dict, repro: dict, results: list[dict]) -> None:
    def agg(frac: float, f) -> tuple[float, float, float]:
        """mean over seeds per draw, then mean / min / max over draws"""
        draws = sorted({r["rep"] for r in results if r["frac"] == frac})
        per = [np.mean([f(r) for r in results if r["frac"] == frac and r["rep"] == d]) for d in draws]
        return float(np.mean(per)), float(np.min(per)), float(np.max(per))

    lines = [
        "# Study A: calibration data size and group-shrunk calibration (development data)",
        "",
        f"Models fixed: M2 seeds {meta['seeds']} of `{meta['run']}`; calibration on V1 {meta['files']['v1']} "
        f"(users subsampled), evaluation on V2 {meta['files']['v2']} G1 groups. Design: docs/DESIGN.md §15.1. "
        "V2 was used to choose M2 and lambda = 100, so the 100% values of lambda = 100 are optimistic.",
        "",
        f"Reproduction check (full V1): {len(repro)} stored V2 values recomputed within {REPRO_TOL:g}.",
        "",
        "Differences are (iii) - (ii); negative = (iii) better. Mean over seeds per draw; brackets = min / max "
        f"over {REPS} draws (one draw at 100%).",
        "",
        "| V1 used | V1 conversions | lambda | macro LogLoss diff | Poisson deviance diff | old median metric diff |",
        "|---:|---:|---:|---|---|---|",
    ]
    for frac in FRACTIONS:
        conv = agg(frac, lambda r: r["v1_conversions"])[0]
        for lam in LAMBDAS:
            k = f"{lam:g}"
            cells = []
            for metric in ("macro_logloss", "poisson_deviance", "old_median_abs_ratio_dev"):
                m, lo, hi = agg(frac, lambda r: r["iii"][k][metric] - r["ii"][metric])
                cells.append(f"{m:+.6f} [{lo:+.6f}, {hi:+.6f}]" if frac < 1 else f"{m:+.6f}")
            lines.append(f"| {frac:.0%} | {conv:,.0f} | {k} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Where the difference comes from (lambda = 100 and lambda = 1)",
        "",
        "Share of the macro LogLoss difference contributed by each group bin (bins by the group's conversions in "
        "the V1 rows used). Groups absent from V1 get the global mapping, so their difference must be 0.",
        "",
        "| V1 used | lambda | " + " | ".join(f"{b} (groups)" for b in BINS) + " |",
        "|---:|---:|" + "---|" * len(BINS),
    ]
    for frac in FRACTIONS:
        for k in ("100", "1"):
            cells = []
            for b in BINS:
                m = agg(frac, lambda r: r["iii"][k]["macro_diff_by_bin"][b])[0]
                n = agg(frac, lambda r: r["groups_per_bin"][b])[0]
                cells.append(f"{m:+.6f} ({n:.0f})")
            lines.append(f"| {frac:.0%} | {k} | " + " | ".join(cells) + " |")
    absent = max(r["iii"][k]["max_abs_diff_absent"] for r in results for k in r["iii"])
    conv = all(r["iii"][k]["converged"] and r["iii"][k]["nonpositive_slopes"] == 0 for r in results for k in r["iii"])
    lines += [
        "",
        f"Checks: largest |difference| in groups absent from V1 = {absent:.2e}; every (iii) fit converged with "
        f"increasing mappings: {conv}.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    t0 = time.time()
    meta = prepare(a.run, a.out)
    print(f"prepared {meta['rows']} in {time.time() - t0:.0f}s", flush=True)
    _init(str(a.out), meta["seeds"])
    repro = reproduce(a.run, meta["seeds"])
    print("reproduction ok", flush=True)
    jobs = [(s, f, r) for f in FRACTIONS for r in (range(1) if f >= 1 else range(REPS)) for s in meta["seeds"]]
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(str(a.out), meta["seeds"])) as ex:
        results = []
        for res in ex.map(task, jobs):
            results.append(res)
            print(f"seed {res['seed']} frac {res['frac']} rep {res['rep']}: {res['seconds']}s", flush=True)
    (a.out / "results.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "reproduction": repro,
                "fractions": FRACTIONS,
                "reps": REPS,
                "lambdas": LAMBDAS,
                "sub_seed": SUB_SEED,
                "results": results,
            },
            indent=1,
        )
    )
    if a.md:
        a.md.parent.mkdir(parents=True, exist_ok=True)
        write_md(a.md, meta, repro, results)
        (a.md.with_suffix(".json")).write_text(
            json.dumps({"meta": meta, "reproduction": repro, "results": results}, indent=1)
        )
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
