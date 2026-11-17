"""Fixed experiment menu for the hypothesis agent, computed by code only.

Split (docs/DESIGN.md §14): models are trained on DISCOVERY files (train-08, 09) and every tool
is evaluated on CONFIRMATION files (train-10, 11), which no earlier study opened. The LLM sees
only first-study evidence (other days); confirmation numbers exist only as tool outputs.

Pre-computed once (`precompute`): LightGBM (first-study settings) on all inputs and with one
feature group removed at a time; predictions and segment flags on the confirmation rows.
Tools (all with user-level bootstrap 95% intervals, B = 500):
    ablation_effect(group, segment)       LogLoss(full) - LogLoss(without group); < 0: the group helps
    association(feature, segment)         CVR(feature present) - CVR(absent)  [associational only]
    interaction(group, segment, segment_b) ablation effect in segment minus in segment_b
    ab_test_design(segment, relative_mde) two-proportion sample size per arm and days of traffic
"""

import json
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm

from .. import groups as G
from .. import vocab as V
from ..features import featurize, read_raw
from ..metrics import bce
from ..models import GBDT
from ..schema import CAT_INPUTS, CTX_COLS, GROUP_KEY, NUM_INPUTS, USER_COLS, VOCAB_COLS

FEATURE_GROUPS = {
    "history_match": ["match_ad_feat0", "match_ad_feat1", "match_ad_feat2", "hist_empty"],
    "visual_similarity": ["mm_feat0_max", "mm_feat0_mean", "mm_feat1_max", "mm_feat1_mean"],
    "user_profile": USER_COLS,
    "context": CTX_COLS,
    "item_attributes": ["ad_feat3", "ad_feat4", "ad_feat5"],
}
NEGATIVE_GROUP = "user_score"  # random noise feature: its true contribution is zero (negative control)
SEGMENTS = ["all", "sparse_G1", "dense", "no_history", "has_history", "same_category_in_history",
            "no_same_category_in_history", "high_visual_similarity", "low_visual_similarity"]
ASSOC_FEATURES = {  # binary feature -> expression on the confirmation frame
    "same_item_in_history": pl.col("match_ad_feat0") > 0,
    "same_category_in_history": pl.col("match_ad_feat1") > 0,
    "same_shop_in_history": pl.col("match_ad_feat2") > 0,
    "high_visual_similarity": pl.col("high_visual"),
    "empty_history": pl.col("hist_empty") == 1,
}
BOOT_REPS, BOOT_SEED = 500, 20261118


def boot_p(reps: np.ndarray) -> float:
    """Two-sided bootstrap p-value for 'estimate = 0', floored at 1 / BOOT_REPS."""
    return float(max(2 * min((reps <= 0).mean(), (reps >= 0).mean()), 1 / BOOT_REPS))


def precompute(config_path: str) -> Path:
    cfg = tomllib.loads(Path(config_path).read_text())
    out = Path("runs") / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    root = Path(cfg["data"]["root"])
    t0 = time.time()
    counts = None
    for f in cfg["data"]["discovery"]:
        frame = read_raw(root / f, columns=VOCAB_COLS).select(VOCAB_COLS)
        counts = V.count_values([frame], counts=counts)
        del frame
    vocab = V.build_vocab(counts)
    disc = pl.concat([featurize(read_raw(root / f), vocab) for f in cfg["data"]["discovery"]])
    conf = pl.concat([featurize(read_raw(root / f), vocab) for f in cfg["data"]["confirmation"]])
    gc = cfg["groups"]
    stats = G.train_stats(disc)
    conf = pl.concat([conf, G.classify(conf, stats, gc["n_min"], gc["c_min"], gc["c_dense"])], how="horizontal")
    q80 = float(disc["mm_feat0_max"].quantile(0.8))  # threshold from DISCOVERY rows only
    conf = conf.with_columns((pl.col("mm_feat0_max") >= q80).alias("high_visual"))
    full = CAT_INPUTS + NUM_INPUTS
    models = {"full": full} | {f"no_{g}": [c for c in full if c not in cols] for g, cols in FEATURE_GROUPS.items()}
    keep = conf.select("user_key", "y", GROUP_KEY, "group_class", "hist_empty", "match_ad_feat0", "match_ad_feat1",
                       "match_ad_feat2", "high_visual")
    # negative control: a random-noise feature ("user_score"), seeded, unrelated to the label by construction
    rng = np.random.default_rng(cfg["seed"] + 7)
    disc = disc.with_columns(pl.Series("noise_score", rng.uniform(size=disc.height)))
    conf = conf.with_columns(pl.Series("noise_score", rng.uniform(size=conf.height)))
    models["full_noise"] = full + ["noise_score"]
    info = {}
    for name, inputs in models.items():
        t = time.time()
        m = GBDT.fit(disc, cfg["lgb"], cfg["seed"], 2000, 50, inputs)
        keep = keep.with_columns(pl.Series(f"p_{name}", m.predict(conf).astype(np.float32)))
        info[name] = {"rounds": m.rounds, "seconds": round(time.time() - t, 1), "inputs": len(inputs)}
        print(f"{name}: {info[name]}", flush=True)
    keep.write_parquet(out / "confirmation_predictions.parquet")
    meta = {"config": cfg, "rows": {"discovery": disc.height, "confirmation": conf.height},
            "discovery_rate": float(disc["y"].mean()), "visual_threshold_q80_discovery": q80, "models": info,
            "seconds": round(time.time() - t0, 1)}
    (out / "precompute.json").write_text(json.dumps(meta, indent=2, default=str))
    return out


class Menu:
    """Loads the pre-computed confirmation predictions and answers tool calls."""

    def __init__(self, folder: Path):
        self.df = pl.read_parquet(folder / "confirmation_predictions.parquet")
        self.meta = json.loads((folder / "precompute.json").read_text())
        users = self.df["user_key"].to_numpy()
        self._uidx = np.unique(users, return_inverse=True)[1]
        self._U = self._uidx.max() + 1
        self.daily_clicks = self.df.height / len(self.meta["config"]["data"]["confirmation"])

    @property
    def _w(self):
        """The same BOOT_REPS user resamples for every tool call (regenerated from a fixed seed)."""
        rng = np.random.default_rng(BOOT_SEED)
        for _ in range(BOOT_REPS):
            yield np.bincount(rng.integers(self._U, size=self._U), minlength=self._U)

    def _seg(self, segment: str) -> np.ndarray:
        d = self.df
        expr = {
            "all": pl.lit(True), "sparse_G1": pl.col("group_class") == "G1", "dense": pl.col("group_class") == "dense",
            "no_history": pl.col("hist_empty") == 1, "has_history": pl.col("hist_empty") == 0,
            "same_category_in_history": pl.col("match_ad_feat1") > 0,
            "no_same_category_in_history": pl.col("match_ad_feat1") == 0,
            "high_visual_similarity": pl.col("high_visual"), "low_visual_similarity": ~pl.col("high_visual"),
        }
        if segment not in expr:
            raise ValueError(f"unknown segment {segment!r}; allowed: {SEGMENTS}")
        return d.with_columns(expr[segment].alias("_seg"))["_seg"].to_numpy()  # with_columns broadcasts literals

    def _boot(self, num: np.ndarray, den: np.ndarray, mask: np.ndarray) -> tuple[float, list[float], np.ndarray]:
        """Ratio of sums sum(num)/sum(den) over `mask` rows, resampling users."""
        n_u = np.bincount(self._uidx[mask], weights=num[mask], minlength=self._U)
        d_u = np.bincount(self._uidx[mask], weights=den[mask], minlength=self._U)
        point = float(n_u.sum() / d_u.sum())
        reps = np.array([(w @ n_u) / (w @ d_u) for w in self._w])
        return point, np.quantile(reps, [0.025, 0.975]).tolist(), reps

    def _loss_diff(self, group: str, mask: np.ndarray) -> tuple[float, list[float], np.ndarray]:
        y = self.df["y"].to_numpy().astype(float)
        if group == NEGATIVE_GROUP:  # with the noise feature minus without it
            d = bce(y, self.df["p_full_noise"].to_numpy()) - bce(y, self.df["p_full"].to_numpy())
        elif group in FEATURE_GROUPS:
            d = bce(y, self.df["p_full"].to_numpy()) - bce(y, self.df[f"p_no_{group}"].to_numpy())
        else:
            raise ValueError(f"unknown feature group {group!r}; allowed: {list(FEATURE_GROUPS)}")
        point, ci, reps = self._boot(d, np.ones_like(d), mask)
        return point, ci, d, reps

    def ablation_effect(self, group: str, segment: str) -> dict:
        m = self._seg(segment)
        point, ci, _, reps = self._loss_diff(group, m)
        return {"tool": "ablation_effect", "group": group, "segment": segment, "rows": int(m.sum()),
                "estimate": point, "ci95": ci, "p_value": boot_p(reps), "meaning": "mean LogLoss(full) - LogLoss(without group); < 0 = group helps"}

    def association(self, feature: str, segment: str) -> dict:
        if feature not in ASSOC_FEATURES:
            raise ValueError(f"unknown feature {feature!r}; allowed: {list(ASSOC_FEATURES)}")
        m = self._seg(segment)
        f = self.df.select(ASSOC_FEATURES[feature]).to_series().to_numpy()
        y = self.df["y"].to_numpy().astype(float)
        a = float(y[m & f].mean())
        b = float(y[m & ~f].mean())
        # interval of the difference with the same user resamples
        ya, yb = (np.bincount(self._uidx[m & f], weights=y[m & f], minlength=self._U),
                  np.bincount(self._uidx[m & ~f], weights=y[m & ~f], minlength=self._U))
        na, nb = (np.bincount(self._uidx[m & f], minlength=self._U), np.bincount(self._uidx[m & ~f], minlength=self._U))
        reps = np.array([(w @ ya) / (w @ na) - (w @ yb) / (w @ nb) for w in self._w])
        return {"tool": "association", "feature": feature, "segment": segment, "rows_with": int((m & f).sum()),
                "rows_without": int((m & ~f).sum()), "cvr_with": a, "cvr_without": b, "estimate": a - b,
                "ci95": np.quantile(reps, [0.025, 0.975]).tolist(), "p_value": boot_p(reps),
                "meaning": "CVR(feature) - CVR(no feature); associational, NOT causal"}

    def interaction(self, group: str, segment: str, segment_b: str) -> dict:
        ma, mb = self._seg(segment), self._seg(segment_b)
        _, _, d, _ = self._loss_diff(group, ma)
        sa, ca_ = np.bincount(self._uidx[ma], weights=d[ma], minlength=self._U), np.bincount(self._uidx[ma], minlength=self._U)
        sb, cb_ = np.bincount(self._uidx[mb], weights=d[mb], minlength=self._U), np.bincount(self._uidx[mb], minlength=self._U)
        point = float(sa.sum() / ca_.sum() - sb.sum() / cb_.sum())
        reps = np.array([(w @ sa) / (w @ ca_) - (w @ sb) / (w @ cb_) for w in self._w])
        return {"tool": "interaction", "group": group, "segment": segment, "segment_b": segment_b,
                "estimate": point, "ci95": np.quantile(reps, [0.025, 0.975]).tolist(), "p_value": boot_p(reps),
                "meaning": "ablation effect in segment minus in segment_b; < 0 = group helps MORE in segment"}

    def ab_test_design(self, segment: str, relative_mde: float, alpha: float = 0.05, power: float = 0.8) -> dict:
        if not 0 < relative_mde < 1:
            raise ValueError("relative_mde must be in (0, 1)")
        m = self._seg(segment)
        p1 = float(self.df["y"].to_numpy()[m].mean())
        p2 = p1 * (1 + relative_mde)
        z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
        n = int(np.ceil(z ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2))
        seg_daily = m.mean() * self.daily_clicks
        return {"tool": "ab_test_design", "segment": segment, "baseline_cvr": p1, "relative_mde": relative_mde,
                "alpha": alpha, "power": power, "clicks_per_arm": n,
                "days_at_50_50_split": float(2 * n / seg_daily) if seg_daily > 0 else None,
                "meaning": "two-proportion z-test sample size; needed to test a CAUSAL claim with a randomized experiment"}


if __name__ == "__main__":
    if sys.argv[1] == "precompute":
        print(precompute(sys.argv[2]))
