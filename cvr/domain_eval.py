"""Study C evaluation (docs/DESIGN.md §16.3, §16.5–16.7). No PyTorch or LightGBM here.

- domain classes from the TRAINING rows only (§16.3): small (share < 1%, clicks >= 3,000), tiny (share < 1%,
  clicks < 3,000), large (share >= 1%), unseen (no training row);
- the primary metric (macro LogLoss over small domains) and the secondary metrics of §16.5;
- one seed-paired user bootstrap that resamples users ONCE per replicate for every statistic and seed pair;
- the V2 selection rule with its tie rule (§16.6), the power check (§16.7) and the two-tier verdict (§16.6).
"""

import numpy as np
import polars as pl
from scipy import sparse

from . import metrics as M
from .schema import DOMAIN_KEY, UNKNOWN

SMALL_SHARE = 0.01
SMALL_MIN_CLICKS = 3000
CLASSES = ["small", "tiny", "large", "unseen"]
TIE = 1e-6
PREFERENCE = ["G-shared", "N-shared", "N-head", "N-star"]  # tie order: lower ladder step first, GBDT before NN
INT_OF = {"G-shared": "G-int", "N-shared": "N-int", "N-head": "N-int", "N-star": "N-int"}
ADOPT_REL = 0.003  # relative improvement point estimate (project-internal adoption criterion)
ADOPT_LARGE = 1e-4  # upper 95% bound of the large-domain click-weighted LogLoss difference
ADOPT_P99_MS = 1.0
POWER_FACTOR = (1.96 + 0.84) / 1.96  # effect detected with 80% power under a normal approximation, two-sided 95%
SEEDS = [2026, 2027, 2028]  # fixed in §16.4; the verdict requires evidence from exactly these seeds
HOLDOUT_FILES = ("train-13.parquet", "train-14.parquet", "test.parquet")  # never in a development config
ROLES = ("train", "v1", "v2")


# ---------------------------------------------------------------- configuration guard (review of §16)
def check_development_sources(data: dict) -> None:
    """Refuse a development `[data]` section that lists a holdout file, uses one file in two roles, or (smoke mode)
    overlapping user buckets. Before revision 4 only a `test` key was refused. Paths are joined with `root` and
    resolved before any comparison, so `./train-11.parquet` and `train-11.parquet` are the same file; holdout names
    are compared without case, and two existing files are compared with os.path.samefile, because this machine's
    file system is case-insensitive (review: `train-11.parquet` and `TRAIN-11.PARQUET` passed as two files)."""
    import os
    from pathlib import Path

    def real(f: str) -> Path:
        return (Path(data.get("root", ".")) / f).resolve()

    def same(a: Path, b: Path) -> bool:
        if a.exists() and b.exists():
            return os.path.samefile(a, b)
        return str(a).casefold() == str(b).casefold()

    holdout = {h.casefold() for h in HOLDOUT_FILES}

    if "test" in data:
        raise ValueError("a development config must not have a `test` entry")
    if data["mode"] == "smoke":
        names = [real(data["smoke_file"])]
        ranges = [tuple(data[f"smoke_{r}_buckets"]) for r in ROLES]
        for i in range(len(ranges)):
            for j in range(i + 1, len(ranges)):
                if max(ranges[i][0], ranges[j][0]) < min(ranges[i][1], ranges[j][1]):
                    raise ValueError(f"smoke buckets of {ROLES[i]} and {ROLES[j]} overlap: {ranges[i]}, {ranges[j]}")
    else:
        names, owner = [], []
        for r in ROLES:
            for f in data[r]:
                rf = real(f)
                for prev, role in owner:
                    if same(rf, prev):
                        raise ValueError(f"{f} ({rf}) is listed for both {role} and {r}")
                owner.append((rf, r))
                names.append(rf)
    bad = [str(f) for f in names if f.name.casefold() in holdout]
    if bad:
        raise ValueError(f"holdout files must not appear in a development config: {bad}")


# ---------------------------------------------------------------- domain classes (§16.3)
def domain_stats(train: pl.DataFrame) -> pl.DataFrame:
    n = train.height
    return (train.group_by(DOMAIN_KEY).agg(pl.len().alias("tr_clicks"), pl.col("y").sum().alias("tr_pos"))
            .with_columns((pl.col("tr_clicks") / n).alias("tr_share")).sort(DOMAIN_KEY))


def domain_class(df: pl.DataFrame, stats: pl.DataFrame, small_share: float = SMALL_SHARE,
                 small_min_clicks: int = SMALL_MIN_CLICKS) -> np.ndarray:
    j = df.select(DOMAIN_KEY).join(stats, on=DOMAIN_KEY, how="left", maintain_order="left")
    cls = (pl.when(pl.col("tr_clicks").is_null() | (pl.col(DOMAIN_KEY) == UNKNOWN)).then(pl.lit("unseen"))
           .when(pl.col("tr_share") >= small_share).then(pl.lit("large"))
           .when(pl.col("tr_clicks") >= small_min_clicks).then(pl.lit("small"))
           .otherwise(pl.lit("tiny")))
    return j.select(cls.alias("c"))["c"].to_numpy()


def domain_summary(stats: pl.DataFrame, small_share: float = SMALL_SHARE,
                   small_min_clicks: int = SMALL_MIN_CLICKS) -> dict:
    c = domain_class(stats.select(DOMAIN_KEY), stats, small_share, small_min_clicks)
    s = stats.with_columns(pl.Series("cls", c))
    out = {"definition": {"key": DOMAIN_KEY, "small_share": small_share, "small_min_clicks": small_min_clicks,
                          "selected_on": "training files only (clicks, never observed positives)"},
           "train_rows": int(stats["tr_clicks"].sum())}
    for k in ("small", "tiny", "large"):
        part = s.filter(pl.col("cls") == k)
        out[k] = {"domains": part.height, "codes": part[DOMAIN_KEY].to_list(),
                  "train_click_share": float(part["tr_share"].sum()), "train_positives": int(part["tr_pos"].sum()),
                  "train_clicks_min": int(part["tr_clicks"].min()) if part.height else None}
    return out


# ---------------------------------------------------------------- metrics (§16.5)
def poisson_deviance(O: np.ndarray, E: np.ndarray) -> np.ndarray:
    O, E = np.asarray(O, float), np.asarray(E, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(O > 0, O * np.log(O / E), 0.0)
    return 2 * (term - (O - E))


def report(y: np.ndarray, p: np.ndarray, dom: np.ndarray, cls: np.ndarray, nonmissing: np.ndarray,
           g1: np.ndarray | None = None, g1_groups: np.ndarray | None = None) -> dict:
    small, large, tiny = cls == "small", cls == "large", cls == "tiny"
    pg = M.per_group(y[small], p[small], dom[small]).sort("g")
    auc_macro, used, excl = M.within_group_auc_macro(y[small], p[small], dom[small]) if small.any() else (None, 0, 0)
    per_dom = M.per_group(y[small | large], p[small | large], dom[small | large]).sort("g")
    out = {
        "small": {"domains": pg.height, "rows": int(small.sum()),
                  "macro_logloss": float(pg["logloss"].mean()) if pg.height else None,
                  "click_weighted_logloss": float(M.bce(y[small], p[small]).mean()) if small.any() else None,
                  "poisson_deviance_mean": float(poisson_deviance(pg["O"], pg["E"]).mean()) if pg.height else None,
                  "pooled_O_over_E": float(y[small].sum() / p[small].sum()) if small.any() else None,
                  "within_domain_auc_macro": auc_macro, "auc_domains_used": used, "auc_domains_excluded": excl},
        "large": {"domains": int(np.unique(dom[large]).size), "rows": int(large.sum()),
                  "click_weighted_logloss": float(M.bce(y[large], p[large]).mean()) if large.any() else None,
                  "pooled_O_over_E": float(y[large].sum() / p[large].sum()) if large.any() else None},
        "tiny": {"domains": int(np.unique(dom[tiny]).size), "rows": int(tiny.sum()), "O": int(y[tiny].sum()),
                 "E": float(p[tiny].sum())},
        "unseen_rows": int((cls == "unseen").sum()),
        "per_domain": per_dom.to_dicts(),
        "overall_nonmissing": M.overall(y[nonmissing], p[nonmissing]),
    }
    if g1 is not None and g1.any():
        out["G1_macro_logloss"] = float(M.per_group(y[g1], p[g1], g1_groups[g1])["logloss"].mean())
    return out


# ---------------------------------------------------------------- weighted within-domain AUC
class WeightedAUC:
    """Per-domain AUC under row weights (e.g. bootstrap multiplicities), ties counted 1/2.
    AUC_d = sum over value levels v of w_pos(v) * (w_neg below v + w_neg(v) / 2) / (W_pos W_neg)."""

    def __init__(self, p: np.ndarray, d: np.ndarray, y: np.ndarray, n_dom: int):
        order = np.lexsort((p, d))
        ds, ps = d[order], p[order]
        new = np.r_[True, (ds[1:] != ds[:-1]) | (ps[1:] != ps[:-1])]
        vid_sorted = np.cumsum(new) - 1
        self.vid = np.empty(p.size, dtype=np.int64)
        self.vid[order] = vid_sorted
        self.v_dom = ds[new]
        self.V, self.D = int(vid_sorted[-1]) + 1 if p.size else 0, n_dom
        self.ypos = y.astype(float)
        first = np.r_[True, self.v_dom[1:] != self.v_dom[:-1]]
        self.first_of = np.full(n_dom, -1)
        self.first_of[self.v_dom[first]] = np.flatnonzero(first)

    def auc(self, w: np.ndarray) -> np.ndarray:
        wp = np.bincount(self.vid, weights=w * self.ypos, minlength=self.V)
        wn = np.bincount(self.vid, weights=w * (1 - self.ypos), minlength=self.V)
        before = np.cumsum(wn) - wn  # negative weight strictly before each level, over all domains
        below = before - before[self.first_of[self.v_dom]]
        num = np.bincount(self.v_dom, weights=wp * (below + 0.5 * wn), minlength=self.D)
        Wp = np.bincount(self.v_dom, weights=wp, minlength=self.D)
        Wn = np.bincount(self.v_dom, weights=wn, minlength=self.D)
        out = np.full(self.D, np.nan)
        ok = (Wp > 0) & (Wn > 0)
        out[ok] = num[ok] / (Wp[ok] * Wn[ok])
        return out


# ---------------------------------------------------------------- paired user bootstrap
def paired_bootstrap(y: np.ndarray, pa: list[np.ndarray], pb: list[np.ndarray], dom: np.ndarray, cls: np.ndarray,
                     users: np.ndarray, reps: int = 1000, seed: int = 0, with_auc: bool = True,
                     pair_seeds: list[int] | None = None) -> dict:
    """A - B for four statistics, seed-paired (A seed k with B seed k), ONE user resample per replicate shared by
    every statistic and seed pair. Models are held fixed: only evaluation-user uncertainty is covered.
    - small_macro: macro LogLoss over small domains (primary); a domain enters a replicate if it has a resampled row
    - small_click: click-weighted LogLoss over small rows; large_click: the same over large rows
    - small_auc_macro: mean over small domains of the per-pair AUC difference, averaged over seed pairs; a domain
      enters if both classes are present in the replicate (the count of such domains is reported)."""
    if len(pa) != len(pb) or not pa:
        raise ValueError("seed-matched prediction lists must be non-empty and of equal length")
    if pair_seeds is not None and len(pair_seeds) != len(pa):
        raise ValueError("pair_seeds must name one seed per prediction pair")
    _, ui = np.unique(users, return_inverse=True)
    U = int(ui.max()) + 1
    small, large = cls == "small", cls == "large"
    codes = np.unique(dom[small])
    gi = np.searchsorted(codes, dom[small])
    G = codes.size
    diff = np.mean([M.bce(y, a) - M.bce(y, b) for a, b in zip(pa, pb)], axis=0)
    per_pair = [M.bce(y, a) - M.bce(y, b) for a, b in zip(pa, pb)]
    ui_s, ui_l = ui[small], ui[large]
    n_small = sparse.csr_matrix((np.ones(small.sum()), (ui_s, gi)), shape=(U, G))

    def by_user(d: np.ndarray) -> sparse.csr_matrix:
        return sparse.csr_matrix((d[small], (ui_s, gi)), shape=(U, G))

    d_mat = by_user(diff)

    def stats(w_user: np.ndarray, d: np.ndarray, d_small: sparse.csr_matrix) -> dict:
        cnt = n_small.T @ w_user
        tot = d_small.T @ w_user
        keep = cnt > 0
        ws, wl = w_user[ui_s], w_user[ui_l]
        return {"small_macro": float((tot[keep] / cnt[keep]).mean()) if keep.any() else np.nan,
                "small_click": float((d[small] * ws).sum() / ws.sum()) if ws.sum() > 0 else np.nan,
                "large_click": float((d[large] * wl).sum() / wl.sum()) if wl.sum() > 0 else np.nan}

    aucs = None
    if with_auc and small.any():
        ys, ds_ = y[small], gi
        aucs = [(WeightedAUC(a[small], ds_, ys, G), WeightedAUC(b[small], ds_, ys, G)) for a, b in zip(pa, pb)]

    def auc_diff(w_rows: np.ndarray) -> tuple[float, int]:
        per = [ia.auc(w_rows) - ib.auc(w_rows) for ia, ib in aucs]
        m = np.mean(per, axis=0)
        ok = np.isfinite(m)
        return (float(m[ok].mean()) if ok.any() else np.nan), int(ok.sum())

    ones = np.ones(U)
    point = stats(ones, diff, d_mat)
    point["per_pair_small_macro"] = [stats(ones, r, by_user(r))["small_macro"] for r in per_pair]
    point["per_pair_seeds"] = list(pair_seeds) if pair_seeds is not None else None
    if aucs is not None:
        point["small_auc_macro"], point["auc_domains"] = auc_diff(np.ones(int(small.sum())))
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in ("small_macro", "small_click", "large_click")} | ({"small_auc_macro": []} if aucs else {})
    for _ in range(reps):
        w = np.bincount(rng.integers(U, size=U), minlength=U).astype(float)
        s = stats(w, diff, d_mat)
        for k in ("small_macro", "small_click", "large_click"):
            draws[k].append(s[k])
        if aucs is not None:
            draws["small_auc_macro"].append(auc_diff(w[ui_s])[0])
    ci = {k: [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))] for k, v in draws.items()}
    return {"point": point, "ci95": ci, "reps": reps, "users": U, "small_domains": G, "seed_pairs": len(pa),
            "note": "A - B; models fixed (evaluation-user uncertainty only); one user resample per replicate"}


def leave_one_domain_out(y: np.ndarray, pa: list[np.ndarray], pb: list[np.ndarray], dom: np.ndarray,
                         cls: np.ndarray) -> dict:
    """Primary difference (seed-pair mean) recomputed with each small domain removed in turn (point estimates)."""
    small = cls == "small"
    diff = np.mean([M.bce(y, a) - M.bce(y, b) for a, b in zip(pa, pb)], axis=0)[small]
    g = pl.DataFrame({"g": dom[small], "d": diff}).group_by("g").agg(pl.col("d").mean()).sort("g")
    vals = g["d"].to_numpy()
    full = float(vals.mean())
    loo = {int(c): float(np.delete(vals, i).mean()) for i, c in enumerate(g["g"].to_list())}
    sign = np.sign(full)
    return {"full": full, "min": min(loo.values()), "max": max(loo.values()),
            "sign_flips": int(sum(np.sign(v) != sign for v in loo.values())), "by_removed_domain": loo}


# ---------------------------------------------------------------- selection, power, verdict (§16.6, §16.7)
def choose_candidate(scores: dict[str, float], tie: float = TIE, preference: list[str] = PREFERENCE) -> str:
    """Lowest score; candidates within `tie` of the best are broken by the preference order."""
    best = min(scores.values())
    near = [c for c in preference if c in scores and scores[c] - best < tie]
    return near[0]


def power_check(h_v2: float, l_int: float) -> dict:
    h_test = h_v2 / np.sqrt(2)
    detectable = POWER_FACTOR * h_test
    threshold = ADOPT_REL * l_int
    return {"h_v2": h_v2, "h_test_optimistic": float(h_test), "detectable_80pct": float(detectable),
            "adoption_effect_abs": float(threshold), "underpowered_expected": bool(detectable > threshold),
            "note": "h_test = h_V2 / sqrt(2) is optimistic: user overlap between the two files, day-to-day variation "
                    "and the three-seed direction rule are not included"}


def verdict(boot: dict, l_int: float, p99_by_seed: dict[int, float] | None) -> dict:
    """Two tiers (§16.6). `boot` is paired_bootstrap(selected, int, pair_seeds=...); `l_int` the int's small macro
    LogLoss; `p99_by_seed` the selected candidate's raw-row p99 (ms) per seed, or None before latency is measured.
    The evidence is checked against the PRE-REGISTERED seeds (SEEDS), never against a list the caller passes
    (review: a run configured with [2026] alone, or a NaN p99, could still give adoption_candidate)."""
    pt, ci = boot["point"], boot["ci95"]
    per, got = pt["per_pair_small_macro"], pt.get("per_pair_seeds")
    if got is None or sorted(got) != sorted(SEEDS) or len(per) != len(SEEDS):
        raise ValueError(f"seed pairs {got} (n={len(per)}) do not match the fixed seeds {SEEDS}")
    if not all(np.isfinite(x) for x in per):
        raise ValueError(f"non-finite per-seed differences {per}")
    if p99_by_seed is not None:
        if sorted(p99_by_seed) != sorted(SEEDS):
            raise ValueError(f"p99 measured for seeds {sorted(p99_by_seed)}, expected {sorted(SEEDS)}")
        bad = {k: v for k, v in p99_by_seed.items() if not (np.isfinite(v) and v > 0)}
        if bad:
            raise ValueError(f"p99 must be finite and positive for every seed: {bad}")
    p99_ms = None if p99_by_seed is None else float(max(p99_by_seed.values()))
    improved = bool(pt["small_macro"] < 0 and ci["small_macro"][1] < 0 and all(x < 0 for x in per))
    rel = float(-pt["small_macro"] / l_int)
    conditions = {"effect_size": {"relative_improvement": rel, "needed": ADOPT_REL, "met": rel >= ADOPT_REL},
                  "large_domains": {"ci95_upper": ci["large_click"][1], "needed_at_most": ADOPT_LARGE,
                                    "met": bool(ci["large_click"][1] <= ADOPT_LARGE)},
                  "latency": {"p99_ms_max_over_seeds": p99_ms, "p99_ms_by_seed": p99_by_seed, "needed_at_most": ADOPT_P99_MS,
                              "met": None if p99_ms is None else bool(p99_ms <= ADOPT_P99_MS)}}
    adopted = improved and all(c["met"] is True for c in conditions.values())
    failed = [k for k, c in conditions.items() if c["met"] is False]
    pending = [k for k, c in conditions.items() if c["met"] is None]
    tier = "adoption_candidate" if adopted else ("improvement_confirmed" if improved else "not_confirmed")
    return {"improvement_confirmed": improved, "tier": tier, "conditions": conditions,
            "failed_conditions": failed, "pending_conditions": pending}


# ---------------------------------------------------------------- study D (docs/DESIGN.md §17)
def paired_bootstrap_groups(y: np.ndarray, pa: list[np.ndarray], pb: list[np.ndarray], groups: np.ndarray,
                            users: np.ndarray, reps: int = 1000, seed: int = 0,
                            pair_seeds: list[int] | None = None) -> dict:
    """A - B on rows that ALL belong to the evaluated groups, seed-paired, one user resample per replicate:
    - macro: group-equal mean of per-group mean LogLoss; click: row-equal mean LogLoss;
    - poisson: mean over groups of 2[O log(O/E) - (O - E)] with O, E the (resampled) positive and predicted sums.
    Poisson deviance is not linear in the rows, so each seed pair's difference is computed and then averaged.
    A group enters a replicate if it has a resampled row."""
    if len(pa) != len(pb) or not pa:
        raise ValueError("seed-matched prediction lists must be non-empty and of equal length")
    _, ui = np.unique(users, return_inverse=True)
    _, gi = np.unique(groups, return_inverse=True)
    U, G = int(ui.max()) + 1, int(gi.max()) + 1
    yf = y.astype(float)

    def mat(v: np.ndarray) -> sparse.csr_matrix:
        return sparse.csr_matrix((v, (ui, gi)), shape=(U, G))

    n_m, y_m = mat(np.ones(y.size)), mat(yf)
    loss = [(M.bce(y, a), M.bce(y, b)) for a, b in zip(pa, pb)]
    la_m = [(mat(a), mat(b)) for a, b in loss]
    pa_m = [(mat(a), mat(b)) for a, b in zip(pa, pb)]

    def stats(w: np.ndarray) -> dict:
        cnt, O = n_m.T @ w, y_m.T @ w
        keep = cnt > 0
        wr = w[ui]
        out = {"macro": [], "click": [], "poisson": []}
        for (la, lb), (ma, mb), (ea, eb) in zip(loss, la_m, pa_m):
            out["macro"].append(float(((ma.T @ w)[keep] / cnt[keep]).mean() - ((mb.T @ w)[keep] / cnt[keep]).mean()))
            out["click"].append(float(((la - lb) * wr).sum() / wr.sum()))
            out["poisson"].append(float(poisson_deviance(O[keep], (ea.T @ w)[keep]).mean()
                                        - poisson_deviance(O[keep], (eb.T @ w)[keep]).mean()))
        return out

    point = stats(np.ones(U))
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in point}
    for _ in range(reps):
        s = stats(np.bincount(rng.integers(U, size=U), minlength=U).astype(float))
        for k in draws:
            draws[k].append(float(np.mean(s[k])))
    return {"point": {k: float(np.mean(v)) for k, v in point.items()},
            "per_pair": point, "per_pair_seeds": list(pair_seeds) if pair_seeds is not None else None,
            "ci95": {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in draws.items()},
            "reps": reps, "users": U, "groups": G, "seed_pairs": len(pa)}
