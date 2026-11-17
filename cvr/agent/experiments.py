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
NEGATIVE_GROUP = "user_score"  # shown to the planner in the negative-control session; backed by noise replicate 0
PLACEBO_REPLICATES = 5
MIN_ROWS, MIN_POSITIVES = 200, 10  # per compared arm; below this a tool returns insufficient_data
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
    """Two-sided bootstrap p-value for 'estimate = 0', floored at 1 / BOOT_REPS and capped at 1.
    Callers must check that every replicate is finite first."""
    return float(min(1.0, max(2 * min((reps <= 0).mean(), (reps >= 0).mean()), 1 / BOOT_REPS)))


def insufficient(tool: str, reason: str, **kw) -> dict:
    return {"tool": tool, "status": "insufficient_data", "reason": reason, **kw}


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
    # placebo / negative control: independent random-noise features, seeded, unrelated to the label.
    # Retraining with noise need not give exactly zero difference (training changes, finite evaluation
    # sample), so several replicates are used to estimate how often the ablation test rejects under the null.
    for r in range(PLACEBO_REPLICATES):
        rng = np.random.default_rng(cfg["seed"] + 100 + r)
        disc = disc.with_columns(pl.Series(f"noise_{r}", rng.uniform(size=disc.height)))
        conf = conf.with_columns(pl.Series(f"noise_{r}", rng.uniform(size=conf.height)))
        models[f"full_noise_{r}"] = full + [f"noise_{r}"]
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
    """Loads the pre-computed confirmation predictions and answers tool calls.

    Every tool checks its compared arms first (rows >= MIN_ROWS and positives >= MIN_POSITIVES, finite
    estimate and replicates). Otherwise it returns status "insufficient_data", which never reaches the
    multiple-testing step.
    """

    def __init__(self, folder: Path):
        self.df = pl.read_parquet(folder / "confirmation_predictions.parquet")
        self.meta = json.loads((folder / "precompute.json").read_text())
        users = self.df["user_key"].to_numpy()
        self._uidx = np.unique(users, return_inverse=True)[1]
        self._U = self._uidx.max() + 1
        self._y = self.df["y"].to_numpy().astype(float)
        self.daily_clicks = self.df.height / len(self.meta["config"]["data"]["confirmation"])

    @property
    def _w(self):
        """The same BOOT_REPS user resamples for every tool call (regenerated from a fixed seed)."""
        rng = np.random.default_rng(BOOT_SEED)
        for _ in range(BOOT_REPS):
            yield np.bincount(rng.integers(self._U, size=self._U), minlength=self._U)

    def _seg(self, segment: str) -> np.ndarray:
        expr = {
            "all": pl.lit(True), "sparse_G1": pl.col("group_class") == "G1", "dense": pl.col("group_class") == "dense",
            "no_history": pl.col("hist_empty") == 1, "has_history": pl.col("hist_empty") == 0,
            "same_category_in_history": pl.col("match_ad_feat1") > 0,
            "no_same_category_in_history": pl.col("match_ad_feat1") == 0,
            "high_visual_similarity": pl.col("high_visual"), "low_visual_similarity": ~pl.col("high_visual"),
        }
        if segment not in expr:
            raise ValueError(f"unknown segment {segment!r}; allowed: {SEGMENTS}")
        return self.df.with_columns(expr[segment].alias("_seg"))["_seg"].to_numpy()  # broadcasts literals

    def _arm_problem(self, name: str, mask: np.ndarray) -> str | None:
        rows, pos = int(mask.sum()), int(self._y[mask].sum())
        if rows < MIN_ROWS or pos < MIN_POSITIVES:
            return f"{name}: {rows} rows, {pos} positives (need >= {MIN_ROWS} rows and >= {MIN_POSITIVES} positives)"
        return None

    def _sums(self, values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (np.bincount(self._uidx[mask], weights=values[mask], minlength=self._U),
                np.bincount(self._uidx[mask], minlength=self._U).astype(float))

    def _finish(self, tool: str, point: float, reps: np.ndarray, extra: dict) -> dict:
        if not (np.isfinite(point) and np.isfinite(reps).all()):
            return insufficient(tool, "non-finite estimate or bootstrap replicates", **extra)
        return {"tool": tool, "status": "ok", **extra, "estimate": float(point),
                "ci95": np.quantile(reps, [0.025, 0.975]).tolist(), "p_value": boot_p(reps)}

    def _loss_delta(self, group: str, replicate: int = 0) -> np.ndarray:
        y = self._y
        if group == NEGATIVE_GROUP:  # with the noise feature minus without it
            return bce(y, self.df[f"p_full_noise_{replicate}"].to_numpy()) - bce(y, self.df["p_full"].to_numpy())
        if group in FEATURE_GROUPS:
            return bce(y, self.df["p_full"].to_numpy()) - bce(y, self.df[f"p_no_{group}"].to_numpy())
        raise ValueError(f"unknown feature group {group!r}; allowed: {list(FEATURE_GROUPS)}")

    def ablation_effect(self, group: str, segment: str, replicate: int = 0) -> dict:
        m = self._seg(segment)
        extra = {"group": group, "segment": segment, "rows": int(m.sum()),
                 "meaning": "mean LogLoss(full) - LogLoss(without group); < 0 = group helps"}
        if (bad := self._arm_problem(segment, m)):
            return insufficient("ablation_effect", bad, **extra)
        s, c = self._sums(self._loss_delta(group, replicate), m)
        reps = np.array([(w @ s) / (w @ c) for w in self._w])
        return self._finish("ablation_effect", s.sum() / c.sum(), reps, extra)

    def association(self, feature: str, segment: str) -> dict:
        if feature not in ASSOC_FEATURES:
            raise ValueError(f"unknown feature {feature!r}; allowed: {list(ASSOC_FEATURES)}")
        m = self._seg(segment)
        f = self.df.with_columns(ASSOC_FEATURES[feature].alias("_f"))["_f"].to_numpy()
        a_m, b_m = m & f, m & ~f
        extra = {"feature": feature, "segment": segment, "rows_with": int(a_m.sum()), "rows_without": int(b_m.sum()),
                 "meaning": "CVR(feature) - CVR(no feature); associational, NOT causal"}
        if (bad := self._arm_problem("with feature", a_m) or self._arm_problem("without feature", b_m)):
            return insufficient("association", bad, **extra)
        ya, na = self._sums(self._y, a_m)
        yb, nb = self._sums(self._y, b_m)
        extra |= {"cvr_with": float(ya.sum() / na.sum()), "cvr_without": float(yb.sum() / nb.sum())}
        reps = np.array([(w @ ya) / (w @ na) - (w @ yb) / (w @ nb) for w in self._w])
        return self._finish("association", extra["cvr_with"] - extra["cvr_without"], reps, extra)

    def interaction(self, group: str, segment: str, segment_b: str) -> dict:
        ma, mb = self._seg(segment), self._seg(segment_b)
        extra = {"group": group, "segment": segment, "segment_b": segment_b,
                 "meaning": "ablation effect in segment minus in segment_b; < 0 = group helps MORE in segment"}
        if segment == segment_b:
            return insufficient("interaction", "segment and segment_b are the same", **extra)
        if (bad := self._arm_problem(segment, ma) or self._arm_problem(segment_b, mb)):
            return insufficient("interaction", bad, **extra)
        d = self._loss_delta(group)
        sa, ca = self._sums(d, ma)
        sb, cb = self._sums(d, mb)
        reps = np.array([(w @ sa) / (w @ ca) - (w @ sb) / (w @ cb) for w in self._w])
        return self._finish("interaction", sa.sum() / ca.sum() - sb.sum() / cb.sum(), reps, extra)

    def ab_test_design(self, segment: str, relative_mde: float, alpha: float = 0.05, power: float = 0.8) -> dict:
        """Two-proportion sample size, inflated by a design effect for clicks clustered within users.

        Approximation: the design effect is the user-bootstrap variance of the segment CVR divided by the
        binomial variance; a real experiment would randomize users and be planned on its own data.
        """
        if not 0 < relative_mde < 1:
            raise ValueError("relative_mde must be in (0, 1)")
        m = self._seg(segment)
        extra = {"segment": segment, "relative_mde": relative_mde, "alpha": alpha, "power": power}
        if (bad := self._arm_problem(segment, m)):
            return insufficient("ab_test_design", bad, **extra)
        y_u, n_u = self._sums(self._y, m)
        p1 = float(y_u.sum() / n_u.sum())
        boot_var = float(np.var([(w @ y_u) / (w @ n_u) for w in self._w]))
        deff = max(1.0, boot_var / (p1 * (1 - p1) / m.sum()))
        p2 = p1 * (1 + relative_mde)
        z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
        n_iid = int(np.ceil(z ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2))
        n = int(np.ceil(n_iid * deff))
        seg_daily = m.mean() * self.daily_clicks
        return {"tool": "ab_test_design", "status": "ok", **extra, "baseline_cvr": p1, "design_effect": deff,
                "clicks_per_arm_iid": n_iid, "clicks_per_arm": n,
                "days_at_50_50_split": float(2 * n / seg_daily) if seg_daily > 0 else None,
                "meaning": "sample size to test a CAUSAL claim with a randomized experiment (approximation)"}

    def placebo_calibration(self, segments: tuple[str, ...] = ("all", "sparse_G1"), alpha: float = 0.05) -> dict:
        """How often the ablation test rejects for pure-noise features (null true), uncorrected alpha."""
        out = []
        for r in range(PLACEBO_REPLICATES):
            for seg in segments:
                res = self.ablation_effect(NEGATIVE_GROUP, seg, replicate=r)
                out.append({"replicate": r, "segment": seg, "status": res["status"],
                            "estimate": res.get("estimate"), "p_value": res.get("p_value")})
        ok = [o for o in out if o["status"] == "ok"]
        return {"tests": len(out), "computable": len(ok), "alpha": alpha,
                "rejections": sum(o["p_value"] < alpha for o in ok),
                "rejections_noise_helps": sum(o["p_value"] < alpha and o["estimate"] < 0 for o in ok), "detail": out}


if __name__ == "__main__":
    if sys.argv[1] == "precompute":
        print(precompute(sys.argv[2]))
