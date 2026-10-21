"""Unit tests for common.py (seeds, priorities, finite check, tie rule, rank, budget, jackknife)."""
import math, itertools
import numpy as np, pytest
import common as C


def test_seed_reproducible_and_keys_distinct():
    a = C.rng(2, 1, 0, 3, 17).integers(0, 2 ** 62, 8); b = C.rng(2, 1, 0, 3, 17).integers(0, 2 ** 62, 8)
    assert (a == b).all()
    keys = [(2, f, s, r, p) for f in (1, 2, 3) for s in (0, 1) for r in range(30) for p in range(0, 476, 7)]
    assert len(set(keys)) == len(keys)
    first = {k: int(C.rng(*k).integers(0, 2 ** 62)) for k in keys}
    assert len(set(first.values())) == len(keys)          # no two keys start the same stream (empirical)


def test_field_encoding_unambiguous():
    assert C._field("1") + C._field("23") != C._field("12") + C._field("3")
    p1 = C.priority("rand", "attack", "A1", "val", 1, "target", 23, "R2", "tr-5")
    p2 = C.priority("rand", "attack", "A1", "val", 12, "target", 3, "R2", "tr-5")
    assert p1 != p2


def test_priority_scope():
    base = dict(stage="attack", fam="A2", split="val", r=4, unit_kind="target", unit=9, rid="tr-1")
    t = C.priority("tie", method="*", **base)
    assert t == C.priority("tie", method="*", **base)                       # deterministic
    r1, r2 = C.priority("rand", method="R2", **base), C.priority("rand", method="B2p", **base)
    assert len({t, r1, r2}) == 3                                           # namespaces / methods separated
    for field, alt in [("stage", "ff"), ("fam", "A3"), ("split", "test"), ("r", 5), ("unit_kind", "control"), ("unit", 10), ("rid", "tr-2")]:
        assert C.priority("rand", method="R2", **{**base, field: alt}) != r1, field
    with pytest.raises(AssertionError):
        C.priority("tie", method="R2", **base)


def test_inj_rid_format():
    assert C.inj_rid("A1", "val", 3, 41, 0, "tr-77") == "inj|v1|attack|A1|val|3|41|0|tr-77"
    with pytest.raises(ValueError):
        C.inj_rid("A1", "val", 3, 41, 0, "tr|77")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_stops(bad):
    with pytest.raises(ValueError):
        C.removal_order([1.0, bad, 0.5], [1, 2, 3], ["a", "b", "c"])


def test_tie_rule_exact_only():
    x = 0.3; y = math.nextafter(x, 1.0)                 # 1 ulp apart, would collapse under round(., 9)
    order = C.removal_order([x, y], [0, 10 ** 18], ["a", "b"])
    assert order == [1, 0]                              # higher raw score first despite worse priority
    order = C.removal_order([x, x, x], [5, 1, 5], ["c", "b", "a"])
    assert order == [1, 2, 0]                           # exact tie -> priority, then rid
    assert C.removal_order([0.0, -0.0], [2, 1], ["a", "b"]) == [1, 0]   # +0 == -0 is a tie


def test_order_invariant_to_input_permutation():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 4, 50).astype(float); rid = [f"tr-{i}" for i in range(50)]
    pr = [C.priority("tie", "attack", "A1", "val", 0, "target", 1, "*", x) for x in rid]
    ref = [rid[i] for i in C.removal_order(s, pr, rid)]
    for _ in range(5):
        perm = rng.permutation(50)
        o = C.removal_order(s[perm], [pr[i] for i in perm], [rid[i] for i in perm])
        assert [rid[perm[i]] for i in o] == ref


def test_ulp_diagnostic():
    x = 1.0; y = math.nextafter(x, 0.0)
    assert C.ulp_near_boundary([x, y], [0, 1], 1) == "near"
    assert C.ulp_near_boundary([1.0, 1.0], [0, 1], 1) == "exact"
    assert C.ulp_near_boundary([1.0, 0.9], [0, 1], 1) == "none"
    assert C.ulp_near_boundary([1.0], [0], 1) == "none"


def test_rank_and_budget():
    others = [5.0, 4.0, 4.0, 3.0]
    assert C.rank_of(6.0, others) == 1 and C.rank_of(4.0, others) == 3.0 and C.rank_of(1.0, others) == 5
    assert C.budget(1, 12, 120) == (12, False) and C.budget(0.5, 5, 120) == (3, False)
    assert C.budget(4, 36, 40) == (66, True)


def test_jackknife_ratio_and_mean():
    rng = np.random.default_rng(1); N = rng.normal(2, 1, 12); M = rng.normal(10, 1, 12)
    th, se, tb, bias = C.jackknife_ratio(N, M)
    assert math.isclose(th, N.sum() / M.sum())
    loo = np.array([(N.sum() - N[i]) / (M.sum() - M[i]) for i in range(12)])
    assert math.isclose(se, math.sqrt(11 / 12 * ((loo - loo.mean()) ** 2).sum()))
    # with constant denominators the jackknife SE equals SD/sqrt(R) of the ratios
    th2, se2, _, _ = C.jackknife_ratio(N, np.full(12, 10.0))
    assert math.isclose(se2, (N / 10).std(ddof=1) / math.sqrt(12), rel_tol=1e-12)
    m, s = C.mean_se([1.0, 2.0, 3.0]); assert m == 2.0 and math.isclose(s, 1 / math.sqrt(3))
