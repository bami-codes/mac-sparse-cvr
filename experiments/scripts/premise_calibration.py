"""E6.3 premise check: does a global model leave group-level probability error in sparse groups?

Rough check inside one training file (users split 70/30 by hash). This is not the
evaluation protocol of the study (see docs/DESIGN.md section 4); it only tests whether
the study's premise holds before building the full pipeline.

For each ad_feat1 group in the holdout we compute O = sum(y), E = sum(p),
V = sum(p(1-p)) and z = (O - E) / sqrt(V). If the model were calibrated within groups
and clicks were independent, var(z) would be close to 1; clustering by user inflates
it somewhat, so we compare var(z) across group bins rather than to 1 alone.

Usage:
    uv run python experiments/scripts/premise_calibration.py data/raw/mac/train/train-17.parquet runs/eda/premise_calibration_train-17.json
"""

import json
import resource
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

PAD = "dbee74ee5fbf"  # inferred padding/missing hash token (EDA E5)
SEED = 2026
CATS = [
    "user_feat0",
    "user_feat1",
    "user_feat2",
    "user_feat3",
    "user_feat4",
    "user_feat5",
    "context_feat1",
    "context_feat2",
    "context_feat3",
    "ad_feat3",
    "ad_feat4",
    "ad_feat5",
    "ad_feat1",
]
SEQ_PAIRS = [("ad_feat0", "ad_feat0_seq"), ("ad_feat1", "ad_feat1_seq"), ("ad_feat2", "ad_feat2_seq")]
MM = ["mm_feat0_seq", "mm_feat1_seq"]


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_row_index("rid")
    out = df.select(
        [
            "rid",
            "user_id",
            (pl.col("last") > 0).cast(pl.Int8).alias("y"),
            (pl.col("ad_feat1") == PAD).alias("is_missing"),
        ]
        + CATS
    )
    for tgt, seq in SEQ_PAIRS:
        m = (
            df.select(["rid", tgt, seq])
            .explode(seq)
            .group_by("rid")
            .agg(
                ((pl.col(seq) == pl.col(tgt)) & (pl.col(seq) != PAD) & (pl.col(tgt) != PAD)).sum().alias(f"match_{tgt}")
            )
        )
        out = out.join(m, on="rid", how="left")
    out = out.join(
        df.select(
            ["rid", (pl.col("ad_feat0_seq").list.n_unique() == 1) & (pl.col("ad_feat0_seq").list.first() == PAD)]
        ).rename({"ad_feat0_seq": "hist_empty"}),
        on="rid",
        how="left",
    )
    mm = df.select(
        ["rid"]
        + [e for c in MM for e in (pl.col(c).list.max().alias(f"{c}_max"), pl.col(c).list.mean().alias(f"{c}_mean"))]
    )
    out = out.join(mm, on="rid", how="left").sort("rid")
    # categorical string -> integer codes by sorted rank (deterministic; Categorical casting
    # can assign codes in thread-dependent order). Codes are per-file; fine for a within-file check.
    out = out.with_columns([(pl.col(c).rank("dense") - 1).cast(pl.Int32).alias(c) for c in CATS])
    return out.with_columns(pl.col("hist_empty").cast(pl.Int8))


def group_calibration(g: pl.DataFrame) -> dict:
    g = g.filter(pl.col("V") > 0).with_columns(((pl.col("O") - pl.col("E")) / pl.col("V").sqrt()).alias("z"))
    big = g.filter(pl.col("E") >= 5)
    # exploratory only, not size-adjusted: tau^2 = (sum (O-E)^2 - sum V) / sum E^2.
    # V uses predicted p, so when true rates differ from p (or clicks are correlated within
    # users) the leftover noise inflates tau much more for small groups. A simulation with the
    # same true relative bias (0.15) gave 0.57 for 50-click groups vs 0.16 for 5,000-click
    # groups. Do not compare bins with it; judge calibration by out-of-sample improvement.
    tau2 = float((((g["O"] - g["E"]) ** 2).sum() - g["V"].sum()) / (g["E"] ** 2).sum()) if g.height else None
    return {
        "n_groups": g.height,
        "mean_E_per_group": float(g["E"].mean()) if g.height else None,
        "n_groups_E_ge5": big.height,
        "exploratory_tau2_not_size_adjusted": tau2,
        "pooled_O_over_E": float(g["O"].sum() / g["E"].sum()) if g.height else None,
        "var_z": float(g["z"].var()) if g.height > 1 else None,
        "share_abs_z_gt2": float((g["z"].abs() > 2).mean()) if g.height else None,
        "median_abs_ratio_dev_E_ge5": float((big["O"] / big["E"] - 1).abs().median()) if big.height else None,
    }


def user_bootstrap_ratio(ho: pl.DataFrame, masks: dict, reps: int = 500, seed: int = 20261117) -> dict:
    """Pooled O/E per mask with a 95% interval from resampling holdout USERS with replacement.

    Exploratory: the fitted model is held fixed, so refitting uncertainty is not included.
    """
    users = ho.select("user_id").unique().sort("user_id").with_row_index("uidx")
    ho = ho.join(users, on="user_id", how="left")
    n_users = users.height
    obs = np.zeros((n_users, len(masks)))
    exp = np.zeros_like(obs)
    out = {}
    for j, (name, mask) in enumerate(masks.items()):
        rows = ho.filter(mask)
        ug = rows.group_by("uidx").agg(pl.col("y").sum().alias("O"), pl.col("p").sum().alias("E"))
        idx = ug["uidx"].to_numpy()
        obs[idx, j], exp[idx, j] = ug["O"].to_numpy(), ug["E"].to_numpy()
        out[name] = {
            "clicks": rows.height,
            "users": ug.height,
            "groups": rows["ad_feat1"].n_unique(),
            "observed": float(obs[:, j].sum()),
            "predicted": float(exp[:, j].sum()),
            "O_over_E": float(obs[:, j].sum() / exp[:, j].sum()),
        }
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(reps):
        w = np.bincount(rng.integers(n_users, size=n_users), minlength=n_users)
        samples.append((w @ obs) / (w @ exp))
    ci = np.quantile(np.asarray(samples), [0.025, 0.975], axis=0)
    for j, name in enumerate(masks):
        out[name]["ci95_user_bootstrap"] = ci[:, j].tolist()
    return out


def main(src: str, out: str) -> None:
    t0 = time.time()
    cols = list({c for c in CATS + ["user_id", "last", "ad_feat0", "ad_feat2"] + [s for _, s in SEQ_PAIRS] + MM})
    feats = build_features(pl.read_parquet(src, columns=cols))
    t_feat = time.time() - t0

    # user-cluster split: 70% train (of which 1/7 used for early stopping), 30% holdout
    h = feats["user_id"].hash(seed=SEED) % 100
    tr_mask = (h < 60).to_numpy()
    es_mask = ((h >= 60) & (h < 70)).to_numpy()
    ho_mask = (h >= 70).to_numpy()
    # is_missing is kept for reporting only (not a model input), so the model matches earlier runs
    xcols = [c for c in feats.columns if c not in ("rid", "user_id", "y", "is_missing")]
    X = feats.select(xcols).to_numpy()
    y = feats["y"].to_numpy()
    cat_idx = [xcols.index(c) for c in CATS]

    params = dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=200,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        seed=SEED,
        num_threads=8,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    dtr = lgb.Dataset(X[tr_mask], y[tr_mask], categorical_feature=cat_idx, free_raw_data=False)
    des = lgb.Dataset(X[es_mask], y[es_mask], categorical_feature=cat_idx, reference=dtr)
    booster = lgb.train(
        params, dtr, num_boost_round=1000, valid_sets=[des], callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    p = booster.predict(X[ho_mask], num_iteration=booster.best_iteration)
    t_train = time.time() - t0 - t_feat

    # Holdout rows with group statistics computed on the TRAIN users only (h < 60).
    # Groups are therefore selected on the 60% train split, not on the whole file.
    train_stats = (
        feats.filter(pl.Series(tr_mask))
        .group_by("ad_feat1")
        .agg(pl.len().alias("tr_clicks"), pl.col("y").sum().alias("tr_pos"))
    )
    ho = (
        feats.filter(pl.Series(ho_mask))
        .select(["user_id", "ad_feat1", "is_missing", "y"])
        .with_columns(pl.Series("p", p))
        .join(train_stats, on="ad_feat1", how="left")
        .with_columns(pl.col("tr_clicks", "tr_pos").fill_null(0))
    )

    nm = ~pl.col("is_missing")
    tc, tp = pl.col("tr_clicks"), pl.col("tr_pos")
    bin_masks = {  # the missing-token group is excluded from every group bin
        "unseen_in_train": nm & (tc == 0),
        "tr_clicks 1-99": nm & (tc > 0) & (tc < 100),
        "tr_clicks>=100,tr_pos<5": nm & (tc >= 100) & (tp < 5),
        "tr_clicks>=100,tr_pos 5-19": nm & (tc >= 100) & tp.is_between(5, 19),
        "tr_clicks>=100,tr_pos 20-49": nm & (tc >= 100) & tp.is_between(20, 49),
        "tr_clicks>=100,tr_pos>=50": nm & (tc >= 100) & (tp >= 50),
        "design_example tr_clicks>=500,tr_pos<20": nm & (tc >= 500) & (tp < 20),
    }

    def per_group(mask: pl.Expr) -> pl.DataFrame:
        return (
            ho.filter(mask)
            .group_by("ad_feat1")
            .agg(
                pl.col("y").sum().alias("O"),
                pl.col("p").sum().alias("E"),
                (pl.col("p") * (1 - pl.col("p"))).sum().alias("V"),
                pl.len().alias("ho_clicks"),
            )
        )

    def overall(mask: np.ndarray) -> dict:
        yy, pp = y[ho_mask][mask], p[mask]
        return {
            "rows": int(mask.sum()),
            "auc": roc_auc_score(yy, pp),
            "logloss": log_loss(yy, pp),
            "O_over_E": float(yy.sum() / pp.sum()),
        }

    missing_np = ho["is_missing"].to_numpy()
    boot_masks = bin_masks | {"missing_group": pl.col("is_missing"), "nonmissing_all": nm, "all_rows": pl.lit(True)}
    report = {
        "file": Path(src).name,
        "note": "within-file user-cluster split; premise check only, not the study protocol",
        "rows": {"train": int(tr_mask.sum()), "early_stop": int(es_mask.sum()), "holdout": int(ho_mask.sum())},
        "lgb_params": params,
        "best_iteration": booster.best_iteration,
        "holdout_overall": overall(np.ones_like(missing_np, dtype=bool)),
        "holdout_overall_nonmissing": overall(~missing_np),
        "group_calibration_by_bin": {
            k: group_calibration(per_group(m)) | {"clicks_share": per_group(m)["ho_clicks"].sum() / ho_mask.sum()}
            for k, m in bin_masks.items()
        },
        "user_bootstrap_O_over_E": user_bootstrap_ratio(ho, boot_masks),
        "seconds": {"features": round(t_feat, 1), "train_predict": round(t_train, 1)},
        "peak_rss_gb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024) / 1e9, 2
        ),  # bytes on macOS, KB on Linux
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
