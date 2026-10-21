"""Small correctness pilot on VALIDATION products only (protocol v4.2). It checks the mechanics, not the
methods: no alpha / S* / R* / R selection is made and no method comparison is summarised.
Pilot settings (mechanics only): alpha = 0.20, rho in {0.5, 1, 2}, R = 2, the first 6 eligible targets per family.
Usage: python pilot.py <sad_dir> <out_dir>"""
import sys, json, time, math
import numpy as np
import common as C, exp as X

SAD, OUT = sys.argv[1], sys.argv[2]
st = X.Setup(f"{SAD}/work"); U = st.universes["val"]
ff = json.load(open(f"{OUT}/ff_matches.json"))
ALPHA, RHOS, R, NT = 0.20, (0.5, 1, 2), 2, 6
checks, fails, times, rows = {}, [], [], []

def check(name, ok, info=""):
    checks.setdefault(name, [0, 0]); checks[name][0 if ok else 1] += 1
    if not ok: fails.append(f"{name}: {info}")

def recompute(T, rem):
    """Independent recomputation of the target metrics from a removal mask."""
    kept = T["rating"][~rem]; s = (kept.sum() + 20 * U.mu) / (len(kept) + 20)
    oth = U.others_clean(T["p"]); rk = lambda x: 1 + (oth > x).sum() + 0.5 * (oth == x).sum()
    return abs(rk(s) - rk(U.clean[T["p"]])), abs(s - U.clean[T["p"]])

for fam in ("A1", "A2", "A3"):
    targets = [p for p in sorted(U.pids) if U.half[p] == C.FAM_HALF[fam]]
    targets = [p for p in targets if len(st.source_pool(fam, p, set(st.norm[U.idx[p]]))) >= math.ceil(ALPHA * U.n[p])][:NT]
    for p in targets:
        for r in range(R):
            T = X.attack(st, U, fam, r, p, ALPHA); T2 = X.attack(st, U, fam, r, p, ALPHA)
            check("attack_reproducible", list(T["rid"]) == list(T2["rid"]) and (T["src"] == T2["src"]).all(), f"{fam} {p} {r}")
            inj = T["inj"]; ninj = T["norm"][inj]; norig = set(T["norm"][~inj])
            check("no_duplicate_injected_text", len(set(ninj)) == T["k"] and not (set(ninj) & norig), f"{fam} {p} {r}")
            parts = [x.split("|") for x in T["rid"][inj]]
            check("inj_rid_format", all(q[:7] == ["inj", "v1", "attack", fam, "val", str(r), str(p)] and q[7] == str(j) and q[8] == st.rid[s]
                                        for j, (q, s) in enumerate(zip(parts, T["src"]))), f"{fam} {p} {r}")
            check("injected_rating", (T["rating"][inj] == C.FAM_RATING[fam]).all() and np.isin(st.rating[T["src"]], C.FAM_SRC_STARS[fam]).all())
            check("source_from_source_products", np.isin(st.pid[T["src"]], st.src_pids).all() and (st.cat[T["src"]] == st.cat[U.idx[p][0]]).all())
            for rho in RHOS:
                b, clipped = C.budget(rho, T["k"], U.n[p])
                t0 = time.perf_counter(); res, rem, sc = X.run_methods(U, T, b, "attack", fam, r, "target"); times.append(time.perf_counter() - t0)
                for m, o in res.items():
                    D, dev = recompute(T, rem[m])
                    check("metric_recompute", math.isclose(D, o["D"], abs_tol=1e-12) and math.isclose(dev, o["score_dev"], abs_tol=1e-12), f"{m} {fam} {p} {r} {rho}")
                    check("n_removed", o["n_removed"] == (0 if m == "B0" else min(b, T["k"]) if m == "LAF" else b), f"{m} {o['n_removed']} vs {b}")
                    check("ranges", (0 <= o["capture"] <= 1) and ((o["L"] == 0) == np.isnan(o["TRR"])) and (np.isnan(o["TRR"]) or 0 <= o["TRR"] <= 1))
                    check("removal_within_target", rem[m].shape[0] == U.n[p] + T["k"])
                    rows.append(dict(fam=fam, p=int(p), r=r, rho=rho, b=b, k=T["k"], method=m, boundary=o["ulp_near"]))
                if b >= T["k"]:
                    check("LAF_restores_reference", res["LAF"]["score_dev"] == 0.0 and res["LAF"]["D"] == 0.0, f"{fam} {p} {r} {rho}")
                n1 = int((T["rating"] == 1).sum())
                check("R2_semantics", (T["rating"][rem["R2"]] == 1).all() if n1 >= b else (rem["R2"][T["rating"] == 1]).all())
                kth = np.sort(T["rating"])[b - 1]
                check("R3_semantics", T["rating"][rem["R3"]].max() <= kth and (T["rating"][~rem["R3"]].min() >= kth))
                check("scores_finite", all(np.isfinite(v).all() for v in sc.values()))
            # order invariance on real data: permute the target's reviews, recompute, compare removed rid sets (rho = 1)
            perm = C.rng(99, C.FAM_NO[fam], p, r).permutation(len(T["rid"]))
            Tp = {k: (v[perm] if isinstance(v, np.ndarray) and len(v) == len(T["rid"]) else v) for k, v in T.items()}
            b, _ = C.budget(1, T["k"], U.n[p]); A, ra, _ = X.run_methods(U, T, b, "attack", fam, r, "target"); B, rb, _ = X.run_methods(U, Tp, b, "attack", fam, r, "target")
            for m in X.ALL_METHODS:
                check("order_invariance", set(T["rid"][ra[m]]) == set(Tp["rid"][rb[m]]), f"{m} {fam} {p} {r}")
# FF: determinism across r for score methods, matched controls of the pilot targets
ffres = []
for fam in ("A1", "A2"):
    targets = [p for p in sorted(U.pids) if U.half[p] == C.FAM_HALF[fam]][:NT]
    for p in targets:
        c = ff[f"val/{p}"]["control"]
        if c is None:
            continue
        Tc = X.clean_target(st, U, c); kc = math.ceil(ALPHA * U.n[c]); b = min(math.ceil(1 * kc), U.n[c] - C.MIN_KEEP)
        r0, rem0, _ = X.run_methods(U, Tc, b, "ff", fam, 0, "control"); r1, rem1, _ = X.run_methods(U, Tc, b, "ff", fam, 1, "control")
        for m in X.SCORE_METHODS:
            check("ff_score_methods_deterministic", (rem0[m] == rem1[m]).all(), f"{m} c={c}")
        ffres.append(dict(fam=fam, target=int(p), control=int(c), b=b, random_methods_differ_across_r={m: bool((rem0[m] != rem1[m]).any()) for m in ("B2p", "R2", "R3")}))
# non-finite score stops the run
try:
    C.removal_order([0.1, float("nan")], [1, 2], ["a", "b"]); check("nan_stops", False)
except ValueError:
    check("nan_stops", True)
summary = dict(settings=dict(split="val", alpha=ALPHA, rhos=RHOS, R=R, targets_per_family=NT, note="mechanics only; no selection"),
               checks={k: dict(passed=v[0], failed=v[1]) for k, v in checks.items()}, failures=fails[:50],
               boundary_diagnostic={c: {m: int(sum(x["boundary"] == c and x["method"] == m for x in rows)) for m in X.ALL_METHODS} for c in ("exact", "near")}, method_runs=len(rows),
               seconds_per_scenario_all_methods=dict(mean=float(np.mean(times)), max=float(np.max(times))), ff=ffres)
json.dump(summary, open(f"{OUT}/pilot_checks.json", "w"), indent=1)
print(json.dumps({k: summary[k] for k in ("checks", "failures", "boundary_diagnostic", "method_runs", "seconds_per_scenario_all_methods")}, indent=1))
