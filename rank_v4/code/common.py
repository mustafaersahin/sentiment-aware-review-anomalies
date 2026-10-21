"""Shared definitions for the v4.2 flagged-product moderation experiment (see ../PROTOCOL_v4.2.md).

Everything that decides which reviews are removed lives here, so that the pre-checks, the pilot and the
later confirmatory run use one implementation: identifiers, hash priorities, seeds, the removal order,
the Bayesian score, the 1-based rank and the replicate-level estimators.
"""
import hashlib, math, unicodedata
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge

ENTROPY = 20261021                      # provisional (protocol section 11)
FAM_NO = {"A1": 1, "A2": 2, "A3": 3}
SPLIT_NO = {"val": 0, "test": 1}
FAM_SRC_STARS = {"A1": (1, 2), "A2": (5,), "A3": (1, 2)}   # star classes of the source texts
FAM_RATING = {"A1": 5.0, "A2": 1.0, "A3": 1.0}              # rating given to injected reviews
FAM_HALF = {"A1": "lower", "A2": "upper", "A3": "upper"}    # half of the split ranking that is attacked
M_PRIOR = 20
ALPHAS = (0.10, 0.20, 0.30)
RHOS = (0.5, 1, 2, 4)
MIN_KEEP = 10
SOURCE_SHARE = 0.30                     # provisional share of reference products held out as text sources
L_CLASSES = ((0, 0), (1, 2), (3, 5), (6, 10 ** 9))


# ---------------------------------------------------------------- seeds, identifiers, priorities
def rng(*key):
    """Generator for a structured SeedSequence key, e.g. rng(2, fam_no, split_no, r, pid)."""
    return np.random.default_rng(np.random.SeedSequence(ENTROPY, spawn_key=tuple(int(k) for k in key)))


def norm_text(t):
    return unicodedata.normalize("NFC", str(t).strip()).casefold()


def _field(s):
    b = s.encode("utf-8")
    return len(b).to_bytes(4, "big") + b


def priority(ns, stage, fam, split, r, unit_kind, unit, method, rid):
    """64-bit keyed BLAKE2b priority of one review in one scenario (protocol section 7).
    ns: 'tie' (shared by all methods, method must be '*') or 'rand' (method-specific)."""
    assert ns in ("tie", "rand") and stage in ("attack", "ff") and unit_kind in ("target", "control")
    assert (method == "*") == (ns == "tie")
    fields = ["v1", ns, stage, fam, split, "-" if r is None else str(int(r)), unit_kind, str(int(unit)), method,
              unicodedata.normalize("NFC", rid)]
    msg = b"".join(_field(f) for f in fields)
    return int.from_bytes(hashlib.blake2b(msg, digest_size=8, key=ENTROPY.to_bytes(8, "big")).digest(), "big")


def inj_rid(fam, split, r, pid, j, src_rid):
    parts = ["inj", "v1", "attack", fam, split, str(int(r)), str(int(pid)), str(int(j)), src_rid]
    if any("|" in x for x in parts):
        raise ValueError(f"'|' inside an injected-rid component: {parts}")
    return "|".join(parts)


# ---------------------------------------------------------------- removal order
def check_finite(s, name="scores"):
    s = np.asarray(s, float)
    bad = ~np.isfinite(s)
    if bad.any():
        raise ValueError(f"{name}: {int(bad.sum())} non-finite value(s) (NaN or +/-inf) - run stopped")


def removal_order(scores, prio, rids):
    """Indices sorted by (-raw score, priority, rid). Raw float64 scores, exact ties only."""
    s = np.asarray(scores, float)
    check_finite(s)
    return sorted(range(len(s)), key=lambda i: (-s[i], prio[i], rids[i]))


def ulp_near_boundary(scores, order, b):
    """Boundary diagnostic (does not affect removal): 'exact' if the scores at positions b-1 and b are equal
    (resolved by the priority), 'near' if 0 < |difference| <= 4*max(ulp), otherwise 'none'."""
    if b <= 0 or b >= len(order):
        return "none"
    x, y = float(scores[order[b - 1]]), float(scores[order[b]])
    if x == y:
        return "exact"
    return "near" if abs(x - y) <= 4 * max(math.ulp(x), math.ulp(y)) else "none"


def budget(rho, k, n_orig):
    """b = ceil(rho*k), capped so that at least MIN_KEEP reviews remain. Returns (b, clipped)."""
    b = math.ceil(rho * k)
    cap = n_orig + k - MIN_KEEP
    return (min(b, cap), b > cap)


# ---------------------------------------------------------------- scores, ranks, robust z
def bayes(ratings, mu, m=M_PRIOR):
    r = np.asarray(ratings, float)
    return (r.sum() + m * mu) / (len(r) + m)


def rank_of(x, others):
    """1-based rank with mid-rank for exact ties: 1 + #{s_q > x} + 0.5 #{s_q == x}."""
    others = np.asarray(others, float)
    return 1.0 + float(np.sum(others > x)) + 0.5 * float(np.sum(others == x))


def rz_params(x):
    m = np.median(x)
    return m, np.median(np.abs(x - m)) * 1.4826 + 1e-9      # as detect_lib.rz


def ctx_distance(X):
    """1 - cosine(e_i, leave-one-out centroid of the product), as detect_lib.py lines 124-130."""
    if len(X) < 3:
        return np.zeros(len(X))
    cen = X.sum(0)
    loo = (cen[None, :] - X) / (len(X) - 1)
    return 1 - (X * loo).sum(1) / (np.linalg.norm(loo, axis=1) + 1e-9)


# ---------------------------------------------------------------- replicate-level estimators
def jackknife_ratio(N, M):
    """theta = sum N / sum M; delete-one-replicate jackknife SE. Returns (theta, se, theta_bar, bias)."""
    N, M = np.asarray(N, float), np.asarray(M, float)
    R = len(N)
    th = N.sum() / M.sum()
    loo = (N.sum() - N) / (M.sum() - M)
    tb = loo.mean()
    se = math.sqrt((R - 1) / R * np.sum((loo - tb) ** 2))
    return th, se, tb, (R - 1) * (tb - th)


def mean_se(x):
    x = np.asarray(x, float)
    return x.mean(), x.std(ddof=1) / math.sqrt(len(x))


def l_class(L):
    for i, (a, b) in enumerate(L_CLASSES):
        if a <= L <= b:
            return i
    raise ValueError(L)


# ---------------------------------------------------------------- frozen decision rules (protocol section 14)
S_CANDIDATES = ("gap_ens", "saad_core", "gap_rm", "gap_cal")   # also the tie order for S*
R_CANDIDATES = ("R2", "R3", "R4")                              # also the tie order for R*
ALPHA_CONFIRM = (0.10, 0.20)
ALPHA_RULE_MIN_NORM_D = 0.08
MAX_EXCLUDED_SHARE = 0.10
MIN_ELIGIBLE = 30
MARGIN_D_REL = 0.10
MCSE_MAX_D_REL = 0.025
MCSE_MAX_K = 0.0125
POLARITY_N = 100
POLARITY_THRESHOLD = 0.80


def _finite(*x):
    return all(np.isfinite(v) for v in x)


def select_r_star(mean_D):
    """R*_A1: rating-only candidate with the smallest raw mean D on A1 validation; ties -> R_CANDIDATES order."""
    c = [m for m in R_CANDIDATES if m in mean_D and _finite(mean_D[m])]
    if not c:
        raise ValueError("no finite rating-only candidate")
    best = min(mean_D[m] for m in c)
    return next(m for m in c if mean_D[m] == best)


def select_s_star(stats):
    """S* on A1 validation. stats = {candidate: (theta_D_rel, theta_K)}, both relative to R*_A1.
    Candidates with theta_D_rel <= 0.10 -> highest theta_K; if none -> smallest theta_D_rel.
    Exact ties -> S_CANDIDATES order. Candidates with non-finite statistics are not eligible."""
    c = [m for m in S_CANDIDATES if m in stats and _finite(*stats[m])]
    if not c:
        raise ValueError("no SAAD candidate with finite statistics")
    ok = [m for m in c if stats[m][0] <= MARGIN_D_REL]
    if ok:
        best = max(stats[m][1] for m in ok)
        return next(m for m in ok if stats[m][1] == best)
    best = min(stats[m][0] for m in c)
    return next(m for m in c if stats[m][0] == best)


def select_alpha(per_alpha):
    """alpha_A1: smallest alpha in ALPHA_CONFIRM with excluded share <= 10% in both splits and validation median
    normalised B0 displacement >= 0.08. per_alpha = {alpha: dict(excl_val, excl_test, median_norm_D_B0_val)}.
    Returns None if no alpha qualifies (then the ranking experiment is exploratory as a whole)."""
    for a in ALPHA_CONFIRM:
        v = per_alpha.get(a)
        if v and v["excl_val"] <= MAX_EXCLUDED_SHARE and v["excl_test"] <= MAX_EXCLUDED_SHARE \
                and v["median_norm_D_B0_val"] >= ALPHA_RULE_MIN_NORM_D:
            return a
    return None


def eligible_targets(n, L, pool, alpha):
    """Targets counted for the K outcome: source pool >= ceil(alpha*n) and L >= 1. FF matching, budget clipping
    and the boundary diagnostic do not affect eligibility. Inputs are aligned sequences; returns a bool list."""
    return [(p >= math.ceil(alpha * nn)) and (l >= 1) for nn, l, p in zip(n, L, pool)]


def h1_decision(d_rel_ci_upper, k_ci_lower):
    """H1_A1 (intersection-union): upper 95% bound of theta_D_rel < 0.10 AND lower 95% bound of theta_K > 0.
    Each bound of the two-sided 95% interval is a one-sided test at 0.025; no multiplicity adjustment."""
    return bool(d_rel_ci_upper < MARGIN_D_REL and k_ci_lower > 0)


def classify_a1(alpha, n_elig_val, n_elig_test, polarity_pass, mcse_d_rel=None, mcse_k=None):
    """'confirmatory' or 'exploratory' with the reasons. mcse_* are the realised test MCSEs (None before the test)."""
    why = []
    if alpha is None: why.append("alpha rule failed")
    if n_elig_val < MIN_ELIGIBLE: why.append(f"validation eligible {n_elig_val} < {MIN_ELIGIBLE}")
    if n_elig_test < MIN_ELIGIBLE: why.append(f"test eligible {n_elig_test} < {MIN_ELIGIBLE}")
    if polarity_pass is not True: why.append("source polarity check not passed")
    if mcse_d_rel is not None and mcse_d_rel > MCSE_MAX_D_REL: why.append("MCSE(theta_D_rel) above threshold")
    if mcse_k is not None and mcse_k > MCSE_MAX_K: why.append("MCSE(theta_K) above threshold")
    return ("exploratory" if why else "confirmatory"), why


def weight_concentration(L):
    """(sum L)^2 / sum L^2 over targets with L >= 1: a weight-concentration diagnostic of the ratio-of-sums TRR.
    It is NOT an independent or effective sample size."""
    L = np.asarray([x for x in L if x >= 1], float)
    return float(L.sum() ** 2 / (L ** 2).sum()) if len(L) else float("nan")


def theta_k_replicate(ret_s, ret_r, L):
    """Primary K in one replicate: ratio-of-sums TRR(S*) - TRR(R*) over targets with L >= 1."""
    ret_s, ret_r, L = (np.asarray(x, float) for x in (ret_s, ret_r, L))
    m = L >= 1
    return float(ret_s[m].sum() / L[m].sum() - ret_r[m].sum() / L[m].sum())


def theta_k_unweighted_replicate(ret_s, ret_r, L):
    """Exploratory sensitivity: mean over targets of TRR_p(S*) - TRR_p(R*), targets with L >= 1 equally weighted."""
    ret_s, ret_r, L = (np.asarray(x, float) for x in (ret_s, ret_r, L))
    m = L >= 1
    return float(np.mean(ret_s[m] / L[m] - ret_r[m] / L[m]))


def polarity_sample(source_rids, fam, n=POLARITY_N):
    """Seeded sample of source-review ids for the human polarity check: ids of the family's source pool, unique,
    in corpus-id order, sampled without replacement with SeedSequence(ENTROPY, (9, fam_no))."""
    ids = sorted(set(source_rids), key=lambda x: int(x.split("-")[1]))
    k = min(n, len(ids))
    return [ids[i] for i in rng(9, FAM_NO[fam]).choice(len(ids), k, replace=False)]


def polarity_result(labels, n_planned=POLARITY_N):
    """labels: one of 'expected', 'not_expected', 'unclear' per sampled review (all must be labelled).
    Share = expected / all sampled (unclear counts as not expected); pass if share >= 0.80."""
    if len(labels) != n_planned or not set(labels) <= {"expected", "not_expected", "unclear"}:
        raise ValueError("polarity labels incomplete or invalid")
    share = sum(x == "expected" for x in labels) / len(labels)
    return share, share >= POLARITY_THRESHOLD


def model_meta(label, model, n_rows):
    """Metadata attached to every result: label, training rows and a hash of the fitted coefficients."""
    w = np.concatenate([np.asarray(model.coef_, "<f8").ravel(), np.asarray([model.intercept_], "<f8")])
    return dict(label=label, train_rows=int(n_rows), alpha=float(model.alpha), sha256=hashlib.sha256(w.tobytes()).hexdigest())


POLARITY_EXPECTED = {"A1": "olumsuz", "A2": "olumlu"}         # A3 uses the A1 source pool


def polarity_map(fam, label):
    """Blind label of the text ('olumlu' / 'olumsuz' / 'belirsiz') -> expected / not_expected / unclear."""
    if label not in ("olumlu", "olumsuz", "belirsiz"):
        raise ValueError(label)
    if label == "belirsiz":
        return "unclear"
    return "expected" if label == POLARITY_EXPECTED[fam] else "not_expected"


def largest_remainder(weights, total):
    """Integer quotas summing to total, proportional to weights (Hamilton); remainder ties -> key order."""
    keys = sorted(weights)
    w = np.array([float(weights[k]) for k in keys]); q = w / w.sum() * total
    base = np.floor(q).astype(int); rem = total - base.sum()
    order = sorted(range(len(keys)), key=lambda i: (-(q[i] - base[i]), keys[i]))
    for i in order[:rem]:
        base[i] += 1
    return {k: int(b) for k, b in zip(keys, base)}


def stratified_unique_sample(candidates, quotas, seen=None):
    """candidates: {stratum: [(rid, norm_text), ...] in seeded draw order}; quotas: {stratum: n}.
    Walks each stratum's order and accepts a review only if its normalised text has not been accepted before
    (within this sample or in `seen`, e.g. the other family). Returns (accepted rids per stratum, rejected count, shortfall)."""
    seen = set() if seen is None else seen
    acc, rej, short = {}, 0, {}
    for s in sorted(quotas):
        acc[s] = []
        for rid, t in candidates.get(s, []):
            if len(acc[s]) == quotas[s]:
                break
            if t in seen:
                rej += 1; continue
            seen.add(t); acc[s].append(rid)
        if len(acc[s]) < quotas[s]:
            short[s] = quotas[s] - len(acc[s])
    return acc, rej, short


# ---------------------------------------------------------------- validation-selection helpers (protocol 14.2)
R_START, R_STEP, R_MAX = 10, 5, 30


def theta_d_rel_parts(D_s, D_r, D_b0):
    """Per-replicate numerators/denominators of theta_D_rel; arrays of shape (R, P) over the same targets.
    N_r = sum_p [D(S) - D(R*)], M_r = sum_p D(B0). Returns (N, M, ratio-of-sums estimate)."""
    D_s, D_r, D_b0 = (np.asarray(x, float) for x in (D_s, D_r, D_b0))
    N = (D_s - D_r).sum(1); M = D_b0.sum(1)
    if M.sum() <= 0:
        raise ValueError("B0 displacement is zero: theta_D_rel undefined")
    return N, M, float(N.sum() / M.sum())


def project_mcse(mcse_val, n_val, n_test):
    """MCSE_test ~ MCSE_val * sqrt(n_val / n_test) (protocol v4.1 B)."""
    return float(mcse_val * math.sqrt(n_val / n_test))


def r_rule_ok(proj_mcse_d_rel, proj_mcse_k):
    return bool(proj_mcse_d_rel <= MCSE_MAX_D_REL and proj_mcse_k <= MCSE_MAX_K)


FROZEN_PROJ_COUNTS = {"D": (32, 64), "K": (30, 56)}     # protocol 14.2.5 (A1, alpha in {0.10, 0.20})


def median_norm_target_mean(D_b0, n_universe):
    """alpha rule statistic (protocol 14.2.5): per-target mean of B0 displacement over the replicates
    (rows = replicates, columns = targets), median over targets, divided by |U| - 1."""
    return float(np.median(np.asarray(D_b0, float).mean(0)) / (n_universe - 1))


# ---------------------------------------------------------------- exploratory analysis helpers (protocol section 16)
def ff_budget(rho, alpha, n_c):
    """FF budget: b_c = ceil(rho * ceil(alpha * n_c)), capped so that MIN_KEEP reviews remain (k = 0)."""
    b = math.ceil(rho * math.ceil(alpha * n_c)); cap = n_c - MIN_KEEP
    return (min(b, cap), b > cap)


def auc(pos, neg):
    """Mann-Whitney AUC: P(score_pos > score_neg) + 0.5 P(equal); NaN if a group is empty."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum(); eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def expected_random_capture(b, k, L1):
    """Expected injected reviews removed when b removals are drawn at random within the 1-star pool (k injected + L1 original)."""
    pool = k + L1
    return min(b, pool) * k / pool if pool else 0.0


def t_interval(x):
    """Mean of replicate values with a t-based 95% interval (descriptive Monte Carlo uncertainty only)."""
    from scipy.stats import t as T
    x = np.asarray(x, float); m, se = mean_se(x)
    h = T.ppf(0.975, len(x) - 1) * se
    return dict(mean=float(m), mcse=float(se), lo=float(m - h), hi=float(m + h))
