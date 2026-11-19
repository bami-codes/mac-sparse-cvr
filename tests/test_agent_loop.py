"""Agent v2 (cvr/agent/loop.py): the scorer finds planted signals and little else, LOND behaves, and a whole run
goes through with a scripted LLM."""

import json

import numpy as np
import polars as pl
import pytest

from cvr.agent import loop as L
from cvr.agent.experiments import FEATURE_GROUPS, NOISE_REPLICATES, Menu


def synthetic_frame(seed: int, signals: set[str], rows: int = 60_000, users: int = 3_000) -> pl.DataFrame:
    """A confirmation frame in the menu's columns. Signals that can be planted:
    'history_in_G1'   the model without history_match is worse on sparse_G1 rows only,
    'visual_in_dense' the model without visual_similarity is worse on dense rows only,
    'category_assoc'  clicks with the same category in history convert more."""
    rng = np.random.default_rng(seed)
    u = rng.integers(0, users, rows)
    cls = rng.choice(["G1", "dense", "other"], rows, p=[0.3, 0.3, 0.4])
    hist_empty = (rng.random(rows) < 0.1).astype(np.int8)
    m0 = np.where(hist_empty == 1, 0, (rng.random(rows) < 0.2).astype(int))
    m1 = np.where(hist_empty == 1, 0, (rng.random(rows) < 0.3).astype(int))
    m2 = np.where(hist_empty == 1, 0, (rng.random(rows) < 0.25).astype(int))
    high = rng.random(rows) < 0.2
    logit = -3.0 + rng.normal(0, 0.8, rows) + (0.9 * (m1 > 0) if "category_assoc" in signals else 0)
    p_true = 1 / (1 + np.exp(-logit))
    y = (rng.random(rows) < p_true).astype(np.int8)
    noisy = lambda s: np.clip(1 / (1 + np.exp(-(logit + rng.normal(0, s, rows)))), 1e-4, 1 - 1e-4)
    p_full = noisy(0.3)
    cols = {"p_full": p_full}
    for g in FEATURE_GROUPS:
        p_no = noisy(0.3)  # a different model of the same quality: no systematic difference
        if g == "history_match" and "history_in_G1" in signals:
            p_no = np.where(cls == "G1", noisy(1.2), p_no)
        if g == "visual_similarity" and "visual_in_dense" in signals:
            p_no = np.where(cls == "dense", noisy(1.2), p_no)
        cols[f"p_no_{g}"] = p_no
    for r in range(NOISE_REPLICATES):
        cols[f"p_full_noise_{r}"] = noisy(0.3)
    return pl.DataFrame({"user_key": u, "y": y, "group_class": cls, "hist_empty": hist_empty, "match_ad_feat0": m0,
                         "match_ad_feat1": m1, "match_ad_feat2": m2, "high_visual": high}
                        | {k: v.astype(np.float32) for k, v in cols.items()})


META = {"config": {"data": {"confirmation": ["synthetic.parquet"]}}}


def passing_for(frame: pl.DataFrame) -> dict[str, int]:
    menu = Menu.from_frame(frame, META)
    res = {}
    for h in L.menu_tests():
        key = L.canon(h)[0]
        name, a = json.loads(key)
        r = menu.ablation_effect(**a, sensitivity=False) if name == "ablation_effect" else getattr(menu, name)(**a)
        res[key] = {k: r.get(k) for k in ("status", "estimate", "ci95", "p_wald")}
    return L.passing_set(res)


def key_of(tool, segment, **kw):
    return L.canon(L.hyp(tool, segment, **kw))[0]


@pytest.mark.parametrize("seed", [1, 2])
def test_one_planted_signal_passes_and_nothing_unrelated_does(seed):
    s = passing_for(synthetic_frame(seed, {"history_in_G1"}))
    assert s.get(key_of("ablation_effect", "sparse_G1", group="history_match")) == -1  # found, full model better
    assert not [k for k in s if '"history_match"' not in k]  # no other group, and no association


def test_several_planted_signals_all_pass():
    s = passing_for(synthetic_frame(3, {"history_in_G1", "visual_in_dense", "category_assoc"}))
    assert s.get(key_of("ablation_effect", "sparse_G1", group="history_match")) == -1
    assert s.get(key_of("ablation_effect", "dense", group="visual_similarity")) == -1
    assert s.get(key_of("association", "all", feature="same_category_in_history")) == 1


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_without_signals_almost_nothing_passes(seed):
    assert len(passing_for(synthetic_frame(seed, set()))) <= 1  # BH over 263 tests


def test_the_bootstrap_p_value_alone_could_not_pass_bh_over_the_menu():
    """Why the scorer uses the Wald p-value: 500 replicates floor the bootstrap p at 0.002, above BH's first cut."""
    from cvr.agent.experiments import BOOT_REPS
    assert 1 / BOOT_REPS > L.Q / len(L.menu_tests())


def test_interaction_order_does_not_change_the_test_or_its_hit():
    a = L.hyp("interaction", "sparse_G1", group="history_match", segment_b="dense")
    b = L.hyp("interaction", "dense", group="history_match", segment_b="sparse_G1")
    (ka, fa), (kb, fb) = L.canon(a), L.canon(b)
    assert ka == kb and fa == -fb
    passing = {ka: -1 if fa == 1 else 1}  # history_match helps more in sparse_G1 than in dense
    assert L.is_hit(a, passing) and not L.is_hit(b, passing)


def test_lond_thresholds():
    lond = L.Lond(0.05)
    beta1 = 0.05 * 6 / np.pi ** 2
    assert lond.test(beta1 * 0.99) and lond.d == 1
    assert not lond.test(0.02)  # second cut: 0.05 * 6 / (pi^2 * 4) * (1 + 1) = 0.0152
    assert lond.i == 2


def test_lond_rejection_formula():
    lond = L.Lond(0.05)
    for i in range(1, 6):
        cut = 0.05 * 6 / (np.pi ** 2 * i ** 2) * (lond.d + 1)
        assert lond.test(cut * 0.999)
    assert lond.d == 5


def test_a_scripted_run_goes_through_and_keeps_the_rules(tmp_path, monkeypatch):
    for d, seed in ((1, 11), (2, 12), (3, 13)):
        folder = tmp_path / f"menu_{d}"
        folder.mkdir()
        synthetic_frame(seed, {"history_in_G1", "category_assoc"}).write_parquet(folder / "confirmation_predictions.parquet")
        (folder / "precompute.json").write_text(json.dumps(META))
    monkeypatch.setattr(L, "MENUS", {d: tmp_path / f"menu_{d}" for d in (1, 2, 3)})
    out = tmp_path / "out"
    out.mkdir()
    items = [{"id": f"E{i}", "title": f"t{i}", "text": "x", "numbers": []} for i in range(1, 7)]
    (out / "evidence.json").write_text(json.dumps({
        "items": items, "association_effects": {"같은 상품": 4.0, "같은 ad_feat1 값": 0.3, "같은 ad_feat2 값": 0.2},
        "visual_top_minus_bottom": 0.5, "empty_minus_has": -0.1}, ensure_ascii=False))
    for d in (1, 2, 3):
        res = {}
        menu = Menu(L.MENUS[d])
        for h in L.menu_tests():
            key = L.canon(h)[0]
            name, a = json.loads(key)
            r = menu.ablation_effect(**a, sensitivity=False) if name == "ablation_effect" else getattr(menu, name)(**a)
            res[key] = {k: r.get(k) for k in ("status", "estimate", "ci95", "p_wald")}
        (out / f"exhaustive_r{d}.json").write_text(json.dumps({"results": res}))
    rec = L.run(scripted=True, out=out)
    L.run_rules(out)
    s = L.score(out, run_file="run_scripted.json")

    for arm in L.ARMS:
        for k, rounds in rec["arms"][arm].items():
            assert len(rounds) == 3
            for d, rows in enumerate(rounds, start=1):
                slots = [r for r in rows if not r.get("control")]
                assert len(slots) <= L.SLOTS
                controls = [r for r in rows if r.get("control")]
                assert len(controls) == (1 if (int(k) == L.NEGATIVE_SESSION and d == 1) else 0)
            # a question asked twice in one session is a repeat the second time, never run again
            keys = [r["key"] for rows in rounds for r in rows if r.get("key") and r["verdict"] != "repeat"]
            assert len(keys) == len(set(keys))
    assert rec["arms"]["H-mem"][1][0] == rec["arms"]["H-fb"][1][0]  # the same round 1
    audit = [json.loads(x) for x in (out / "audit.jsonl").read_text().splitlines()]
    mem = next(a for a in audit if a["role"] == "s1_H-mem_r2")["request"]["user"]
    fb = next(a for a in audit if a["role"] == "s1_H-fb_r2")["request"]["user"]
    shown = ("전환율 차이", "LogLoss 차이")
    assert not any(s in mem for s in shown) and any(s in fb for s in shown)  # results are shown to H-fb only
    assert any(a["role"] == "s1_H-fb-think_r2_think" for a in audit)  # two-step calls for the reasoning arm
    for name in ("evidence_rule", "adaptive_rule"):
        assert all(len(r) == L.SLOTS for r in json.loads((out / "rules.json").read_text())[name])
    assert set(s["feedback_pairs_fb_minus_mem"]) == {"R2", "R3"}
    assert 0 <= s["baselines"]["random"]["expected_hits_per_round"][0] <= L.SLOTS


def test_v21_repairs_bad_slots_and_lets_feedback_be_cited(tmp_path, monkeypatch):
    from cvr.agent.session import validate
    for d, seed in ((1, 21), (2, 22), (3, 23)):
        folder = tmp_path / f"menu_{d}"
        folder.mkdir()
        synthetic_frame(seed, {"category_assoc"}).write_parquet(folder / "confirmation_predictions.parquet")
        (folder / "precompute.json").write_text(json.dumps(META))
    monkeypatch.setattr(L, "MENUS", {d: tmp_path / f"menu_{d}" for d in (1, 2, 3)})
    monkeypatch.setattr(L, "SESSIONS", 2)
    out = tmp_path / "out"
    out.mkdir()
    items = [{"id": f"E{i}", "title": f"t{i}", "text": "x", "numbers": []} for i in range(1, 7)]
    (out / "evidence.json").write_text(json.dumps({"items": items}, ensure_ascii=False))
    rec = L.run(scripted=True, out=out, v21=True)
    assert rec["version"] == "v2.1" and rec["repairs"]  # the scripted plans repeat questions, so repairs happen
    audit = [json.loads(x) for x in (out / "audit_v21.jsonl").read_text().splitlines()]
    assert any(a["role"].endswith("_repair") for a in audit)
    fb = next(a for a in audit if a["role"] == "s1_H-fb_r2")["request"]["user"]
    assert "[R1-1]" in fb
    # an id of an earlier round is accepted as evidence; an unknown one is not
    h = L.normalize(L.hyp("ablation_effect", "all", group="context", evidence_ids=("R1-2",)))
    assert not validate(h, {"E1"} | L.history_ids([[{}, {}]]))
    assert validate(h, {"E1"})
    # slot problems are found before anything runs
    bad = [L.hyp("association", "high_visual_similarity", feature="high_visual_similarity",
                 expected="higher_cvr_with_feature"),
           L.hyp("ablation_effect", "all", group="context"), L.hyp("ablation_effect", "all", group="context")]
    probs = L.slot_problems(bad, set(), {"E1"})
    assert set(probs) == {0, 2} and "같은 라운드" in probs[2]


class PickFirst:
    """A planner stand-in that keeps the schema it was given and picks the first allowed candidates."""

    def __init__(self, direction_for=None):
        self.schemas, self.direction_for = [], direction_for or {}

    def structured(self, system, user, schema, role, max_tokens=4096):
        self.schemas.append(schema)
        allowed = schema["properties"]["hypotheses"]["items"]["properties"]["candidate"]["enum"]
        return {"hypotheses": [{"candidate": c, "direction": self.direction_for.get(c, "group_helps"), "statement": "",
                                "evidence": ["E1"]} for c in allowed[:L.SLOTS]]}


def test_v22_offers_only_untested_possible_candidates_and_maps_directions():
    items = [{"id": "E1", "title": "t", "text": "x", "numbers": []}]
    menu = L.candidate_menu(1)
    assert len(menu) == len(L.menu_tests()) == 263
    assert all(L.empty_arm_by_definition(h["feature"], h["segment"]) is None for h in menu.values() if h["tool"] == "association")
    tested = {L.canon(menu["C001"])[0], L.canon(menu["C002"])[0]}
    llm = PickFirst()
    hyps = L.plan_round_v22(llm, items, 1, None, False, "t", tested)
    allowed = llm.schemas[0]["properties"]["hypotheses"]["items"]["properties"]["candidate"]["enum"]
    assert "C001" not in allowed and "C002" not in allowed and len(allowed) == 261
    assert [h["candidate"] for h in hyps] == allowed[:5]
    # an interaction picked as helping more in segment_b becomes the reversed test
    cid = next(c for c, h in menu.items() if h["tool"] == "interaction")
    h = L.from_candidate({"candidate": cid, "direction": "helps_more_in_segment_b", "evidence": ["E1"]}, menu)
    assert (h["segment"], h["segment_b"]) == (menu[cid]["segment_b"], menu[cid]["segment"])
    assert h["expected"] == "group_helps_more_in_segment" and L.canon(h)[1] == -1
    # a direction that does not fit the tool stays visible as an invalid hypothesis
    from cvr.agent.session import validate
    bad = L.from_candidate({"candidate": "C001", "direction": "higher_cvr_with_feature", "evidence": ["E1"]}, menu)
    assert validate(L.normalize(bad), {"E1"})
    assert "N001" in L.candidate_menu(L.NEGATIVE_SESSION) and "N001" not in menu


def test_v22_scripted_run_goes_through(tmp_path, monkeypatch):
    for d, seed in ((1, 31), (2, 32), (3, 33)):
        folder = tmp_path / f"menu_{d}"
        folder.mkdir()
        synthetic_frame(seed, {"category_assoc"}).write_parquet(folder / "confirmation_predictions.parquet")
        (folder / "precompute.json").write_text(json.dumps(META))
    monkeypatch.setattr(L, "MENUS", {d: tmp_path / f"menu_{d}" for d in (1, 2, 3)})
    monkeypatch.setattr(L, "SESSIONS", 2)
    out = tmp_path / "out"
    out.mkdir()
    (out / "evidence.json").write_text(json.dumps({"items": [{"id": f"E{i}", "title": "t", "text": "x", "numbers": []}
                                                             for i in range(1, 7)]}))
    rec = L.run(scripted=True, out=out, v22=True)
    assert rec["version"] == "v2.2"
    rows = [r for arm in rec["arms"].values() for rounds in arm.values() for rr in rounds for r in rr]
    assert not [r for r in rows if r["verdict"] == "invalid" and "impossible by definition" in "; ".join(r["issues"])]
