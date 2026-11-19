"""Fixed experiment menu for the hypothesis agent, computed by code only.

Split (docs/DESIGN.md §14, files set in the config): models are trained on DISCOVERY files and every
tool is evaluated on CONFIRMATION files. The LLM sees only first-study evidence (other days);
confirmation numbers exist only as tool outputs.

Pre-computed once (`precompute`): LightGBM on all inputs and with one feature group removed at a time;
predictions and segment flags on the confirmation rows.
Tools (all with paired user-level bootstrap 95% intervals, B = 500):
    ablation_effect(group, segment)       LogLoss(full) - LogLoss(without group); < 0: the full model is better
    association(feature, segment)         CVR(feature present) - CVR(absent)  [associational only]
    interaction(group, segment, segment_b) ablation effect in segment minus in segment_b
    ab_test_design(segment, relative_mde) two-proportion sample size per arm and days of traffic

Estimand of the model comparisons (ablation_effect, interaction): the mean confirmation LogLoss
difference between the TWO MODELS THAT WERE TRAINED, under fixed training conditions. The bootstrap
resamples confirmation users with the fitted models held fixed (inference conditional on the fit, as in
LOCO, Lei et al. 2018 §6). It does not cover other training runs or other days, and it is not a test
of whether a feature carries conversion information in general.
Noise models (an uninformative extra feature, same training conditions) are a TRAINING SENSITIVITY
analysis: they show how much held-out LogLoss moves when training changes without new information.
They are reported next to each comparison and never enter a verdict.
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
NOISE_REPLICATES = 5
# Per compared arm; below these a tool returns insufficient_data (the operating gate).
# - 312 positives: a PROVISIONAL threshold from one synthetic configuration (docs/results/agent_bootstrap_check.md:
#   with fewer, the model-comparison bootstrap rejected 6.6-7.4% at 5%). Not a general sample-size guarantee.
# - 2 users and 2 converting users: blocks only the obviously degenerate cases of a USER bootstrap (with one user
#   every resample is identical: zero-width interval, p at its floor). Users, converting users and concentration
#   are reported for every arm; no claim is made that these floors are sufficient.
MIN_ROWS, MIN_POSITIVES, MIN_USERS = 200, 312, 2
SEGMENTS = ["all", "sparse_G1", "dense", "no_history", "has_history", "same_category_in_history",
            "no_same_category_in_history", "high_visual_similarity", "low_visual_similarity"]
ASSOC_FEATURES = {  # binary feature -> expression on the confirmation frame
    "same_item_in_history": pl.col("match_ad_feat0") > 0,
    "same_category_in_history": pl.col("match_ad_feat1") > 0,
    "same_shop_in_history": pl.col("match_ad_feat2") > 0,
    "high_visual_similarity": pl.col("high_visual"),
    "empty_history": pl.col("hist_empty") == 1,
}
# What a segment fixes, as (condition, value), and the condition each association feature tests. Used to reject
# association requests whose comparison is impossible by definition (one arm is empty whatever the data), which
# the final run reported as "insufficient data" (feature = segment, or empty_history inside has_history).
SEGMENT_FIXES = {"no_history": ("empty_history", True), "has_history": ("empty_history", False),
                 "same_category_in_history": ("same_category_in_history", True),
                 "no_same_category_in_history": ("same_category_in_history", False),
                 "high_visual_similarity": ("high_visual_similarity", True),
                 "low_visual_similarity": ("high_visual_similarity", False)}
# implications that follow from the feature code itself (not from the data): an empty history is all padding and
# matching skips padding, so no click with an empty history has the same item in its history (match_ad_feat0
# compares the item sequence that defines hist_empty). Other matches use other sequences, so they are left to the
# data gate.
IMPLIED_BY = {("empty_history", True): [("same_item_in_history", False)]}


def empty_arm_by_definition(feature: str, segment: str) -> str | None:
    """Why association(feature, segment) has an arm that is empty by definition, or None."""
    fixed = SEGMENT_FIXES.get(segment)
    if fixed is None:
        return None
    for cond, value in [fixed] + IMPLIED_BY.get(fixed, []):
        if cond == feature:
            arm = "without_feature" if value else "with_feature"
            return f"segment {segment} fixes {feature} = {value}, so the {arm} arm is empty by definition"
    return None


BOOT_REPS, BOOT_SEED = 500, 20261118
TEST_MODEL_COMPARISON = "paired user bootstrap, fitted models held fixed"


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
    # training sensitivity / negative control: independent random-noise features, seeded, unrelated to the
    # label. A model retrained with one of them is a different fitted model, so its held-out LogLoss can
    # really differ from the full model's; the replicates show how large such differences are.
    for r in range(NOISE_REPLICATES):
        rng = np.random.default_rng(cfg["seed"] + 100 + r)
        disc = disc.with_columns(pl.Series(f"noise_{r}", rng.uniform(size=disc.height)))
        conf = conf.with_columns(pl.Series(f"noise_{r}", rng.uniform(size=conf.height)))
        models[f"full_noise_{r}"] = full + [f"noise_{r}"]
    info = {}
    fixed = bool(cfg.get("fixed_rounds"))
    params = cfg["lgb"] | ({"feature_fraction": 1.0, "bagging_fraction": 1.0, "bagging_freq": 0} if fixed else {})
    rounds = None
    for name, inputs in models.items():
        t = time.time()
        if fixed and rounds is not None:  # same budget and deterministic settings for every variant
            m = GBDT.fit_fixed(disc, params, cfg["seed"], rounds, inputs)
        else:
            m = GBDT.fit(disc, params, cfg["seed"], 2000, 50, inputs)
            rounds = m.rounds  # the full model fixes the budget (early stopping on discovery users only)
        keep = keep.with_columns(pl.Series(f"p_{name}", m.predict(conf).astype(np.float32)))
        info[name] = {"rounds": m.rounds, "seconds": round(time.time() - t, 1), "inputs": len(inputs)}
        print(f"{name}: {info[name]}", flush=True)
    keep.write_parquet(out / "confirmation_predictions.parquet")
    meta = {"config": cfg, "rows": {"discovery": disc.height, "confirmation": conf.height},
            "fixed_rounds": fixed, "discovery_rate": float(disc["y"].mean()), "visual_threshold_q80_discovery": q80, "models": info,
            "seconds": round(time.time() - t0, 1)}
    (out / "precompute.json").write_text(json.dumps(meta, indent=2, default=str))
    return out


class Menu:
    """Loads the pre-computed confirmation predictions and answers tool calls.

    Every tool checks its compared arms first (rows >= MIN_ROWS and positives >= MIN_POSITIVES, finite
    estimate and replicates). Otherwise it returns status "insufficient_data", which never reaches the
    multiple-testing step.
    """

    def __init__(self, folder: Path, **gate):
        self._setup(pl.read_parquet(folder / "confirmation_predictions.parquet"),
                    json.loads((folder / "precompute.json").read_text()), **gate)

    @classmethod
    def from_frame(cls, df: pl.DataFrame, meta: dict, **gate) -> "Menu":
        """Same tools on an in-memory frame (synthetic checks of the tests, scripts/agent_bootstrap_check.py)."""
        menu = cls.__new__(cls)
        menu._setup(df, meta, **gate)
        return menu

    def _setup(self, df: pl.DataFrame, meta: dict, min_rows: int = MIN_ROWS, min_positives: int = MIN_POSITIVES,
               min_users: int = MIN_USERS):
        self.df, self.meta = df, meta
        self.gate = {"min_rows": min_rows, "min_positives": min_positives, "min_users": min_users,
                     "min_converting_users": min_users}
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

    def _arm_stats(self, mask: np.ndarray) -> dict:
        """Size and concentration of one compared arm, in the unit the bootstrap resamples (users)."""
        clicks = np.bincount(self._uidx[mask], minlength=self._U)
        pos = np.bincount(self._uidx[mask], weights=self._y[mask], minlength=self._U)
        rows, positives = int(clicks.sum()), int(pos.sum())
        return {"rows": rows, "positives": positives, "users": int((clicks > 0).sum()),
                "converting_users": int((pos > 0).sum()),
                "max_user_click_share": float(clicks.max() / rows) if rows else None,
                "max_user_positive_share": float(pos.max() / positives) if positives else None,
                "effective_users": float(rows ** 2 / (clicks.astype(float) ** 2).sum()) if rows else 0.0}

    def _check(self, **arms: np.ndarray) -> tuple[str | None, dict]:
        """Operating gate for every compared arm: (first problem or None, per-arm statistics)."""
        g, stats, problems = self.gate, {}, []
        for name, mask in arms.items():
            st = stats[name] = self._arm_stats(mask)
            if (st["rows"] < g["min_rows"] or st["positives"] < g["min_positives"] or st["users"] < g["min_users"]
                    or st["converting_users"] < g["min_converting_users"]):
                problems.append(f"{name}: {st['rows']} rows, {st['positives']} positives, {st['users']} users, "
                                f"{st['converting_users']} converting users (need >= {g['min_rows']}, "
                                f">= {g['min_positives']}, >= {g['min_users']}, >= {g['min_converting_users']})")
        return (problems[0] if problems else None), stats

    def _sums(self, values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (np.bincount(self._uidx[mask], weights=values[mask], minlength=self._U),
                np.bincount(self._uidx[mask], minlength=self._U).astype(float))

    def _finish(self, tool: str, point: float, reps: np.ndarray, extra: dict) -> dict:
        if not (np.isfinite(point) and np.isfinite(reps).all()):
            return insufficient(tool, "non-finite estimate or bootstrap replicates", **extra)
        if np.ptp(reps) == 0 and point != 0:  # resampling cannot vary (e.g. one user): no interval, no p-value
            return insufficient(tool, "bootstrap replicates do not vary", **extra)
        return {"tool": tool, "status": "ok", **extra, "estimate": float(point),
                "ci95": np.quantile(reps, [0.025, 0.975]).tolist(), "p_value": boot_p(reps)}

    def _loss_delta(self, group: str, replicate: int = 0) -> np.ndarray:
        y = self._y
        if group == NEGATIVE_GROUP:  # with the noise feature minus without it
            return bce(y, self.df[f"p_full_noise_{replicate}"].to_numpy()) - bce(y, self.df["p_full"].to_numpy())
        if group in FEATURE_GROUPS:
            return bce(y, self.df["p_full"].to_numpy()) - bce(y, self.df[f"p_no_{group}"].to_numpy())
        raise ValueError(f"unknown feature group {group!r}; allowed: {list(FEATURE_GROUPS)}")

    def _point(self, delta: np.ndarray, m: np.ndarray) -> float:
        s, c = self._sums(delta, m)
        return float(s.sum() / c.sum())

    def _boot(self, delta: np.ndarray, m: np.ndarray) -> tuple[float, np.ndarray]:
        """Paired user bootstrap of a mean row difference: the fitted models stay fixed, users are resampled."""
        s, c = self._sums(delta, m)
        return float(s.sum() / c.sum()), np.array([(w @ s) / (w @ c) for w in self._w])

    def _sensitivity(self, segment: str, segment_b: str | None = None, exclude: int | None = None) -> dict:
        """The same comparison for 'full + one noise feature' vs 'full', per noise replicate (point estimates).

        Descriptive only: how far held-out LogLoss moves when training changes without new information.
        It is not a null distribution and never enters a verdict: a normal-reference test built from 4-5 such
        replicates rejected 8-9% of the time at a nominal 5% on a synthetic null."""
        ma = self._seg(segment)
        mb = self._seg(segment_b) if segment_b else None
        est = []
        for r in range(NOISE_REPLICATES):
            if r == exclude:  # the negative control is described by the OTHER noise replicates
                continue
            d = self._loss_delta(NEGATIVE_GROUP, r)
            a = self._point(d, ma)
            est.append(a - self._point(d, mb) if mb is not None else a)
        return {"replicates": len(est), "estimates": est, "max_abs": float(np.max(np.abs(est))),
                "meaning": "same comparison, full + one noise feature vs full (same training conditions); "
                           "descriptive, not used in the verdict"}

    def ablation_effect(self, group: str, segment: str, replicate: int = 0, sensitivity: bool = True) -> dict:
        m = self._seg(segment)
        bad, arms = self._check(**{segment: m})
        extra = {"group": group, "segment": segment, "rows": int(m.sum()), "arms": arms,
                 "meaning": "mean LogLoss(full) - LogLoss(without group) of the two fitted models; "
                            "< 0 = the full model is better on these confirmation rows"}
        if bad:
            return insufficient("ablation_effect", bad, **extra)
        point, reps = self._boot(self._loss_delta(group, replicate), m)
        out = self._finish("ablation_effect", point, reps, extra | {"test": TEST_MODEL_COMPARISON})
        if out["status"] == "ok" and sensitivity:
            sens = self._sensitivity(segment, exclude=replicate if group == NEGATIVE_GROUP else None)
            out |= {"training_sensitivity": sens, "beyond_sensitivity_range": abs(point) > sens["max_abs"]}
        return out

    def association(self, feature: str, segment: str) -> dict:
        if feature not in ASSOC_FEATURES:
            raise ValueError(f"unknown feature {feature!r}; allowed: {list(ASSOC_FEATURES)}")
        m = self._seg(segment)
        f = self.df.with_columns(ASSOC_FEATURES[feature].alias("_f"))["_f"].to_numpy()
        a_m, b_m = m & f, m & ~f
        bad, arms = self._check(with_feature=a_m, without_feature=b_m)
        extra = {"feature": feature, "segment": segment, "rows_with": int(a_m.sum()), "rows_without": int(b_m.sum()),
                 "arms": arms, "meaning": "CVR(feature) - CVR(no feature); associational, NOT causal"}
        if bad:
            return insufficient("association", bad, **extra)
        ya, na = self._sums(self._y, a_m)
        yb, nb = self._sums(self._y, b_m)
        extra |= {"cvr_with": float(ya.sum() / na.sum()), "cvr_without": float(yb.sum() / nb.sum())}
        reps = np.array([(w @ ya) / (w @ na) - (w @ yb) / (w @ nb) for w in self._w])
        return self._finish("association", extra["cvr_with"] - extra["cvr_without"], reps, extra)

    def interaction(self, group: str, segment: str, segment_b: str) -> dict:
        ma, mb = self._seg(segment), self._seg(segment_b)
        extra = {"group": group, "segment": segment, "segment_b": segment_b,
                 "meaning": "ablation effect in segment minus in segment_b (same two fitted models); "
                            "< 0 = the full model gains MORE in segment"}
        if segment == segment_b:
            return insufficient("interaction", "segment and segment_b are the same", **extra)
        bad, extra["arms"] = self._check(**{segment: ma, segment_b: mb})
        if bad:
            return insufficient("interaction", bad, **extra)
        d = self._loss_delta(group)
        sa, ca = self._sums(d, ma)
        sb, cb = self._sums(d, mb)
        reps = np.array([(w @ sa) / (w @ ca) - (w @ sb) / (w @ cb) for w in self._w])  # same users in both arms
        point = float(sa.sum() / ca.sum() - sb.sum() / cb.sum())
        out = self._finish("interaction", point, reps, extra | {"test": TEST_MODEL_COMPARISON})
        if out["status"] == "ok":
            sens = self._sensitivity(segment, segment_b)
            out |= {"training_sensitivity": sens, "beyond_sensitivity_range": abs(point) > sens["max_abs"]}
        return out

    def ab_test_design(self, segment: str, relative_mde: float, alpha: float = 0.05, power: float = 0.8) -> dict:
        """Two-proportion sample size, inflated by a design effect for clicks clustered within users.

        Approximation: the design effect is the user-bootstrap variance of the segment CVR divided by the
        binomial variance; a real experiment would randomize users and be planned on its own data.
        """
        if not 0 < relative_mde < 1:
            raise ValueError("relative_mde must be in (0, 1)")
        m = self._seg(segment)
        bad, arms = self._check(**{segment: m})
        extra = {"segment": segment, "relative_mde": relative_mde, "alpha": alpha, "power": power, "arms": arms}
        if bad:
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

    def training_sensitivity(self, segments: tuple[str, ...] = ("all", "sparse_G1"), alpha: float = 0.05) -> dict:
        """Each noise model vs the full model, with the same fixed-model bootstrap as every comparison.

        A row with p < alpha is a real difference between two fitted models, not a false positive of the test:
        it shows that an uninformative change to training alone moves held-out LogLoss that much. The error
        rate of the test itself is checked on synthetic data with a known answer (scripts/agent_bootstrap_check.py).
        """
        out = []
        for r in range(NOISE_REPLICATES):
            for seg in segments:
                res = self.ablation_effect(NEGATIVE_GROUP, seg, replicate=r, sensitivity=False)
                out.append({"replicate": r, "segment": seg, "status": res["status"], "estimate": res.get("estimate"),
                            "ci95": res.get("ci95"), "p_value": res.get("p_value")})
        ok = [o for o in out if o["status"] == "ok"]
        return {"comparisons": len(out), "computable": len(ok), "alpha": alpha,
                "different_at_alpha": sum(o["p_value"] < alpha for o in ok),
                "noise_model_better_at_alpha": sum(o["p_value"] < alpha and o["estimate"] < 0 for o in ok),
                "max_abs_estimate": max((abs(o["estimate"]) for o in ok), default=None),
                "note": "noise model minus full model, uncorrected alpha; describes training sensitivity, "
                        "not the false-positive rate of the test",
                "detail": out}


if __name__ == "__main__":
    if sys.argv[1] == "precompute":
        print(precompute(sys.argv[2]))
