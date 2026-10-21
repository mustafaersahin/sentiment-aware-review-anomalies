"""Unit tests (synthetic inputs) for the frozen decision rules of protocol section 14."""
import math
import numpy as np, pytest
from sklearn.linear_model import Ridge
import common as C


def test_s_star_margin_then_k():
    st = {"gap_ens": (0.12, 0.30), "saad_core": (0.08, 0.05), "gap_rm": (0.02, 0.10), "gap_cal": (0.10, 0.10)}
    assert C.select_s_star(st) == "gap_rm"            # within margin (0.10 inclusive), highest K; gap_rm before gap_cal
    st["gap_rm"] = (0.02, 0.09)
    assert C.select_s_star(st) == "gap_cal"           # 0.10 counts as within the margin


def test_s_star_fallback_and_ties():
    st = {"gap_ens": (0.20, 0.5), "saad_core": (0.15, 0.1), "gap_rm": (0.15, 0.9), "gap_cal": (0.30, 0.0)}
    assert C.select_s_star(st) == "saad_core"         # none within margin -> smallest D_rel, tie -> fixed order
    st = {m: (0.05, 0.2) for m in C.S_CANDIDATES}
    assert C.select_s_star(st) == "gap_ens"


def test_s_star_nonfinite_and_empty():
    st = {"gap_ens": (float("nan"), 0.9), "saad_core": (0.05, 0.1)}
    assert C.select_s_star(st) == "saad_core"
    with pytest.raises(ValueError):
        C.select_s_star({"gap_ens": (float("nan"), 0.0)})
    with pytest.raises(ValueError):
        C.select_s_star({"R2": (0.0, 0.0)})          # rating-only methods are not S* candidates


def test_r_star():
    assert C.select_r_star({"R2": 5.0, "R3": 3.0, "R4": 4.0}) == "R3"
    assert C.select_r_star({"R2": 3.0, "R3": 3.0, "R4": 3.0}) == "R2"
    assert C.select_r_star({"R2": float("nan"), "R4": 1.0, "B2p": 0.0}) == "R4"


def test_alpha_rule():
    good = dict(excl_val=0.0, excl_test=0.0, median_norm_D_B0_val=0.09)
    assert C.select_alpha({0.10: good, 0.20: good}) == 0.10
    assert C.select_alpha({0.10: {**good, "median_norm_D_B0_val": 0.07}, 0.20: good}) == 0.20
    assert C.select_alpha({0.10: {**good, "excl_test": 0.11}, 0.20: {**good, "median_norm_D_B0_val": 0.0799}}) is None
    assert C.select_alpha({0.30: good}) is None       # 0.30 is not a confirmatory alpha


def test_eligibility_counting():
    n = [40, 120, 100, 50]; L = [0, 3, 1, 2]; pool = [100, 25, 20, 10]
    assert C.eligible_targets(n, L, pool, 0.10) == [False, True, True, True]     # L=0 excluded
    assert C.eligible_targets(n, L, pool, 0.20) == [False, True, True, True]     # k = 8, 24, 20, 10: pool 10 >= 10
    assert C.eligible_targets(n, L, pool, 0.30) == [False, False, False, False]  # k = 36, 30, 15


def test_h1_and_classification():
    assert C.h1_decision(0.099, 0.001) and not C.h1_decision(0.10, 0.5) and not C.h1_decision(0.0, 0.0)
    assert C.classify_a1(0.1, 30, 56, True) == ("confirmatory", [])
    s, why = C.classify_a1(0.1, 29, 56, True); assert s == "exploratory" and "validation" in why[0]
    assert C.classify_a1(None, 30, 56, True)[0] == "exploratory"
    assert C.classify_a1(0.1, 30, 56, None)[0] == "exploratory"           # polarity check not done yet
    assert C.classify_a1(0.1, 30, 56, True, mcse_d_rel=0.026, mcse_k=0.01)[0] == "exploratory"
    assert C.classify_a1(0.1, 30, 56, True, mcse_d_rel=0.025, mcse_k=0.0125)[0] == "confirmatory"


def test_weight_concentration_and_k():
    assert C.weight_concentration([2, 2, 2, 0]) == pytest.approx(3.0)
    assert C.weight_concentration([60, 1, 1]) == pytest.approx(62 ** 2 / 3602)
    assert math.isnan(C.weight_concentration([0, 0]))
    L = [4, 1, 0]; rs = [4, 0, 0]; rr = [2, 1, 0]
    assert C.theta_k_replicate(rs, rr, L) == pytest.approx(4 / 5 - 3 / 5)
    assert C.theta_k_unweighted_replicate(rs, rr, L) == pytest.approx(((1 - 0.5) + (0 - 1)) / 2)


def test_polarity_sample_and_result():
    ids = [f"tr-{i}" for i in (5, 3, 3, 900, 12, 7, 41)]
    a = C.polarity_sample(ids, "A1", n=4); b = C.polarity_sample(list(reversed(ids)), "A1", n=4)
    assert a == b and len(set(a)) == 4 and set(a) <= set(ids)               # order-independent, no repeats
    assert C.rng(9, 1).integers(0, 2 ** 62) != C.rng(9, 2).integers(0, 2 ** 62)   # A1 and A2 use separate streams
    assert len(C.polarity_sample(ids, "A1", n=100)) == 6                    # pool smaller than n -> whole pool
    lab = ["expected"] * 80 + ["unclear"] * 5 + ["not_expected"] * 15
    assert C.polarity_result(lab) == (0.80, True)
    assert C.polarity_result(["expected"] * 79 + ["unclear"] * 21) == (0.79, False)
    with pytest.raises(ValueError):
        C.polarity_result(["expected"] * 99)


def test_model_meta():
    X = np.random.default_rng(0).normal(size=(50, 3)); y = X @ [1.0, -2.0, 0.5]
    m1 = C.model_meta("ridge_srcout", Ridge(alpha=10.0).fit(X, y), 50)
    m2 = C.model_meta("ridge_srcout", Ridge(alpha=10.0).fit(X, y), 50)
    m3 = C.model_meta("ridge_srcout", Ridge(alpha=10.0).fit(X[:40], y[:40]), 40)
    assert m1 == m2 and m1["sha256"] != m3["sha256"] and m1["train_rows"] == 50 and m1["alpha"] == 10.0


def test_polarity_map():
    assert [C.polarity_map("A1", x) for x in ("olumsuz", "olumlu", "belirsiz")] == ["expected", "not_expected", "unclear"]
    assert [C.polarity_map("A2", x) for x in ("olumsuz", "olumlu", "belirsiz")] == ["not_expected", "expected", "unclear"]
    with pytest.raises(ValueError):
        C.polarity_map("A1", "neutral")


def test_largest_remainder():
    q = C.largest_remainder({"a": 1, "b": 1, "c": 1}, 100)
    assert sum(q.values()) == 100 and q == {"a": 34, "b": 33, "c": 33}
    assert C.largest_remainder({"x": 3, "y": 1}, 10) == {"x": 8, "y": 2}


def test_stratified_unique_sample():
    cand = {"a": [("r1", "t1"), ("r2", "t1"), ("r3", "t3")], "b": [("r4", "t3"), ("r5", "t5")]}
    acc, rej, short = C.stratified_unique_sample(cand, {"a": 2, "b": 2})
    assert acc == {"a": ["r1", "r3"], "b": ["r5"]} and rej == 2 and short == {"b": 1}
    seen = {"t5"}
    acc, rej, short = C.stratified_unique_sample({"b": [("r5", "t5"), ("r6", "t6")]}, {"b": 1}, seen)
    assert acc == {"b": ["r6"]} and rej == 1 and "t6" in seen


def test_exploratory_helpers():
    assert C.ff_budget(1, 0.2, 120) == (24, False) and C.ff_budget(4, 0.3, 40) == (30, True) and C.ff_budget(4, 0.3, 30) == (20, True)
    assert C.auc([3, 2], [1, 2]) == pytest.approx((2 + 1 + 0.5) / 4) and math.isnan(C.auc([], [1]))
    assert C.expected_random_capture(2, 2, 2) == pytest.approx(1.0) and C.expected_random_capture(10, 2, 2) == pytest.approx(2.0)
    ti = C.t_interval([1.0, 2.0, 3.0]); assert ti["mean"] == 2.0 and ti["lo"] < 2.0 < ti["hi"]
