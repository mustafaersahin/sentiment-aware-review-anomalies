"""Validation selection run for A1 (protocol section 14.2): alpha_A1 -> R*_A1 -> S* -> R, frozen to a record.

Test-split separation: only the validation Universe is built; exp.attack / exp.run_methods are wrapped so that
any call with a non-validation universe raises; every call is counted by split and the counts are written to the
record. From test products only pre-attack counts are read (n_p, L_p, source-pool size, clean half).

Frozen definitions (protocol 14.2.5): alpha median = median over pool-ok A1 validation targets of the per-target
mean B0 displacement over r = 0..9, divided by |U|-1; MCSE projection outcome-specific with the frozen counts
D 32 -> 64, K 30 -> 56 (checked against the pre-attack counts, the run stops if they differ); alpha, R*, S* are
selected with r = 0..9 only, additional replicates serve only the MCSE/R rule.
Usage: python select_validation.py <sad_dir> <out_json>"""
import sys, json, math, time, hashlib, platform, argparse, datetime
import numpy as np, pandas as pd, sklearn
import common as C, exp as X

CALLS = {"attack": {}, "run_methods": {}}


def guard(fn, name):
    def wrapped(*a, **k):
        U = a[1] if name == "attack" else a[0]
        CALLS[name][U.split] = CALLS[name].get(U.split, 0) + 1
        if U.split != "val":
            raise PermissionError(f"{name} called with split {U.split!r}: only validation is allowed in the selection run")
        return fn(*a, **k)
    return wrapped


X.attack = guard(X.attack, "attack")
X.run_methods = guard(X.run_methods, "run_methods")


def preattack_counts(st, split, fam, alpha):
    """Pre-attack eligibility of a split without building its Universe: clean half, n_p, L_p, pool size."""
    pids = st.split_pids[split]
    idx = {p: np.where(st.pid == p)[0] for p in pids}
    mu = np.concatenate([st.rating[idx[p]] for p in pids]).mean()
    clean = {p: C.bayes(st.rating[idx[p]], mu) for p in pids}
    order = sorted(pids, key=lambda p: (-clean[p], p)); h = len(order) // 2
    half = {p: ("upper" if i < h else "lower") for i, p in enumerate(order)}
    T = sorted(p for p in pids if half[p] == C.FAM_HALF[fam])
    n = [len(idx[p]) for p in T]; L = [int(np.sum(st.rating[idx[p]] <= 2)) for p in T]
    pool = [len(st.source_pool(fam, p, set(st.norm[idx[p]]))) for p in T]
    ok = [q >= math.ceil(alpha * nn) for q, nn in zip(pool, n)]
    elig = C.eligible_targets(n, L, pool, alpha)
    return dict(targets=len(T), pool_ok=int(sum(ok)), excluded_share=float(1 - np.mean(ok)), eligible=int(sum(elig)))


def run_reps(st, U, targets, alpha, reps, methods):
    """D, retained_low and L per (replicate, target) for the given methods at rho = 1 (validation only)."""
    out = {m: {"D": np.zeros((len(reps), len(targets))), "ret": np.zeros((len(reps), len(targets)))} for m in methods}
    L = np.array([U.L[p] for p in targets])
    for i, r in enumerate(reps):
        for j, p in enumerate(targets):
            T = X.attack(st, U, "A1", r, p, alpha)
            b, _ = C.budget(1, T["k"], U.n[p])
            res, _, _ = X.run_methods(U, T, b, "attack", "A1", r, "target", methods=methods)
            for m in methods:
                out[m]["D"][i, j] = res[m]["D"]; out[m]["ret"][i, j] = res[m]["retained_low"]
    return out, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sad"); ap.add_argument("out")
    a = ap.parse_args()
    t0 = datetime.datetime.now(datetime.timezone.utc).isoformat()
    st = X.Setup(f"{a.sad}/work", splits=("val",))
    assert set(st.universes) == {"val"}
    U = st.universes["val"]
    rec = dict(started_utc=t0, definitions=dict(alpha_median="target_mean (protocol 14.2.5)", proj_counts=f"outcome_specific {C.FROZEN_PROJ_COUNTS}",
                                                selection_replicates="r = 0..9"), steps={})
    pol = json.load(open(f"{a.sad}/rank_v4/out/polarity_result.json"))
    polarity_pass = bool(pol["A1"]["passed"])
    reps0 = list(range(C.R_START))
    # step 1: alpha
    per_alpha, b0_cache = {}, {}
    for alpha in C.ALPHA_CONFIRM:
        v = preattack_counts(st, "val", "A1", alpha); t = preattack_counts(st, "test", "A1", alpha)
        elig_val, excl_val = X.eligibility(st, U, "A1", alpha)
        assert v["eligible"] == len(elig_val)
        targets = sorted(p for p in U.pids if U.half[p] == "lower" and p not in excl_val)
        o, _ = run_reps(st, U, targets, alpha, reps0, ("B0",))
        med = C.median_norm_target_mean(o["B0"]["D"], len(U.pids))
        per_alpha[alpha] = dict(excl_val=v["excluded_share"], excl_test=t["excluded_share"], median_norm_D_B0_val=med,
                                eligible_val=v["eligible"], eligible_test=t["eligible"], pool_ok_val=v["pool_ok"], pool_ok_test=t["pool_ok"],
                                targets_val=len(targets))
        b0_cache[alpha] = targets
    alpha = C.select_alpha(per_alpha)
    rec["steps"]["alpha"] = dict(per_alpha={str(k): v for k, v in per_alpha.items()}, selected=alpha, rule="smallest alpha in {0.10,0.20}: excl<=0.10 both splits and median >= 0.08")
    if alpha is None:
        rec["classification"] = C.classify_a1(None, 0, 0, polarity_pass)
        return finish(a, st, rec)
    targets = b0_cache[alpha]; ev, et = per_alpha[alpha]["eligible_val"], per_alpha[alpha]["eligible_test"]
    # step 2-3: R* and S* with R0 replicates
    meths = ("B0",) + C.R_CANDIDATES + C.S_CANDIDATES
    o, L = run_reps(st, U, targets, alpha, reps0, meths)
    meanD = {m: float(o[m]["D"].mean()) for m in C.R_CANDIDATES}
    rstar = C.select_r_star(meanD)
    stats = {}
    for m in C.S_CANDIDATES:
        _, _, d_rel = C.theta_d_rel_parts(o[m]["D"], o[rstar]["D"], o["B0"]["D"])
        k = float(np.mean([C.theta_k_replicate(o[m]["ret"][i], o[rstar]["ret"][i], L) for i in range(len(reps0))]))
        stats[m] = (d_rel, k)
    sstar = C.select_s_star(stats)
    rec["steps"]["r_star"] = dict(mean_D=meanD, selected=rstar)
    rec["steps"]["s_star"] = dict(stats={m: dict(theta_D_rel=v[0], theta_K=v[1]) for m, v in stats.items()}, selected=sstar)
    # step 4: R with incremental replicates (S*, R*, B0 only)
    keep = {m: o[m] for m in ("B0", rstar, sstar)}
    nD_val, nD_test = len(targets), per_alpha[alpha]["pool_ok_test"]
    nK_val = int(sum(U.L[p] >= 1 for p in targets)); nK_test = et
    if ((nD_val, nD_test), (nK_val, nK_test)) != (C.FROZEN_PROJ_COUNTS["D"], C.FROZEN_PROJ_COUNTS["K"]):
        raise SystemExit(f"projection counts {(nD_val, nD_test, nK_val, nK_test)} differ from the frozen {C.FROZEN_PROJ_COUNTS}: stop")
    R, path, chosen = C.R_START, [], None
    while True:
        N, M, d_rel = C.theta_d_rel_parts(keep[sstar]["D"], keep[rstar]["D"], keep["B0"]["D"])
        _, se_d, _, _ = C.jackknife_ratio(N, M)
        kr = [C.theta_k_replicate(keep[sstar]["ret"][i], keep[rstar]["ret"][i], L) for i in range(R)]
        k, se_k = C.mean_se(kr)
        pd_, pk = C.project_mcse(se_d, nD_val, nD_test), C.project_mcse(se_k, nK_val, nK_test)
        path.append(dict(R=R, theta_D_rel=d_rel, mcse_D_rel_val=se_d, theta_K=k, mcse_K_val=se_k, proj_mcse_D_rel=pd_, proj_mcse_K=pk, ok=C.r_rule_ok(pd_, pk)))
        if path[-1]["ok"]:
            chosen = R; break
        if R >= C.R_MAX:
            break
        more, _ = run_reps(st, U, targets, alpha, list(range(R, R + C.R_STEP)), ("B0", rstar, sstar))
        keep = {m: {q: np.vstack([keep[m][q], more[m][q]]) for q in ("D", "ret")} for m in keep}
        R += C.R_STEP
    rec["steps"]["R"] = dict(path=path, selected=chosen, R_used=R, projection_counts=dict(D=[nD_val, nD_test], K=[nK_val, nK_test]))
    status, why = C.classify_a1(alpha, ev, et, polarity_pass)
    if chosen is None:
        status, why = "exploratory", why + ["R rule not met at R_max"]
    rec["classification"] = (status, why)
    rec["frozen_for_test"] = dict(alpha_A1=alpha, R_star=rstar, S_star=sstar, R=chosen if chosen else C.R_MAX, status=status)
    finish(a, st, rec)


def finish(a, st, rec):
    sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
    rec["polarity_check"] = json.load(open(f"{a.sad}/rank_v4/out/polarity_result.json"))
    rec["polarity_check"] = {f: {k: rec["polarity_check"][f][k] for k in ("share_expected", "passed")} for f in ("A1", "A2")}
    rec["access_evidence"] = dict(universes_built=sorted(st.universes), calls_by_split=CALLS,
                                  test_reads="pre-attack counts only (preattack_counts: n_p, L_p, pool size, clean half)")
    rec["hashes"] = dict(code={f: sha(f"{a.sad}/rank_v4/code/{f}") for f in ("select_validation.py", "common.py", "exp.py")},
                         data={f: sha(f"{a.sad}/{f}") for f in ("work/tr_corpus.parquet", "work/emb_tr.npy", "work/sent_tr.npy")},
                         protocol=sha(f"{a.sad}/rank_v4/PROTOCOL_v4.2.md"), frozen_metadata=sha(f"{a.sad}/rank_v4/out/frozen_metadata.json"),
                         polarity_result=sha(f"{a.sad}/rank_v4/out/polarity_result.json"))
    rec["model"] = st.ridge_meta
    rec["env"] = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, entropy=C.ENTROPY)
    rec["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump(rec, open(a.out, "x"), indent=1, default=float)      # 'x': never overwrite an existing record
    print(json.dumps({k: rec[k] for k in ("definitions", "steps", "classification", "access_evidence") if k in rec}, indent=1, default=float))


if __name__ == "__main__":
    main()
