"""Tests for the validation-selection driver: helpers (synthetic), the test-split guard, and (integration, validation
data only, read-only) agreement of the pre-attack counting path with exp.eligibility."""
import os, math, types
import numpy as np, pytest
import common as C
import select_validation as SV
import exp as X

SAD = os.path.join(os.path.dirname(__file__), "..", "..")


def test_theta_d_rel_parts():
    Ds = [[2, 4], [1, 1]]; Dr = [[1, 1], [1, 2]]; Db = [[10, 10], [5, 5]]
    N, M, th = C.theta_d_rel_parts(Ds, Dr, Db)
    assert list(N) == [4, -1] and list(M) == [20, 10] and th == pytest.approx(3 / 30)
    with pytest.raises(ValueError):
        C.theta_d_rel_parts([[1]], [[0]], [[0]])


def test_projection_and_r_rule():
    assert C.project_mcse(0.04, 32, 64) == pytest.approx(0.04 / math.sqrt(2))
    assert C.r_rule_ok(0.025, 0.0125) and not C.r_rule_ok(0.0251, 0.0) and not C.r_rule_ok(0.0, 0.0126)
    assert (C.R_START, C.R_STEP, C.R_MAX) == (10, 5, 30)


def test_guard_blocks_test_split():
    fake = types.SimpleNamespace(split="test")
    with pytest.raises(PermissionError):
        X.attack(None, fake, "A1", 0, 1, 0.1)
    with pytest.raises(PermissionError):
        X.run_methods(fake, {}, 1, "attack", "A1", 0, "target")
    assert SV.CALLS["attack"].get("test", 0) >= 1 and SV.CALLS["run_methods"].get("test", 0) >= 1


@pytest.mark.skipif(not os.path.exists(os.path.join(SAD, "work", "tr_corpus.parquet")), reason="data not available")
def test_preattack_counts_match_eligibility_on_validation():
    st = X.Setup(os.path.join(SAD, "work"), splits=("val",))
    assert set(st.universes) == {"val"}
    for a in C.ALPHA_CONFIRM:
        v = SV.preattack_counts(st, "val", "A1", a)
        el, ex = X.eligibility(st, st.universes["val"], "A1", a)
        assert v["eligible"] == len(el) and v["pool_ok"] == v["targets"] - len(ex)


def test_median_norm_target_mean():
    D = [[0, 10, 4], [2, 30, 4]]          # 2 replicates x 3 targets -> target means 1, 20, 4 -> median 4
    assert C.median_norm_target_mean(D, 65) == pytest.approx(4 / 64)
    assert C.FROZEN_PROJ_COUNTS == {"D": (32, 64), "K": (30, 56)}
