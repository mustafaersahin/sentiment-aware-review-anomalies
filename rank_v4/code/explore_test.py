"""EXPLORATORY test analysis (protocol section 16). No alpha / S* / R* / R selection, no winner, no confirmatory test.
Before running, it checks that the frozen plan record exists, that its sha256 matches the stored hash and that this
script and its modules match the hashes listed in the record; otherwise it stops.
Usage: python explore_test.py <sad_dir> <plan_record_json> <rows_csv_gz> <summary_json>"""
import sys, json, math, hashlib, datetime
import numpy as np, pandas as pd
import common as C, exp as X

SAD, PLAN, ROWS, SUMMARY = sys.argv[1:5]
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
stored = open(PLAN + ".sha256").read().split()[0]
if sha(PLAN) != stored:
    raise SystemExit("plan record hash mismatch: stop")
plan = json.load(open(PLAN))
for f, h in plan["code_sha256"].items():
    if sha(f"{SAD}/rank_v4/code/{f}") != h:
        raise SystemExit(f"code {f} differs from the frozen plan record: stop")
P = plan["parameters"]
SPLIT = P["split"]
st = X.Setup(f"{SAD}/work", splits=(SPLIT,))
assert set(st.universes) == {SPLIT}
U = st.universes[SPLIT]
ff = json.load(open(f"{SAD}/rank_v4/out/ff_matches.json"))
REPS = list(range(P["R"]))
rows, ffrows = [], []


def removed_counts(T, rem):
    orig = ~T["inj"]
    d = {f"n{c}": int(np.sum(orig & (T["rating"] == c))) for c in (1, 2, 3, 4, 5)}
    d.update({f"rm{c}": int(np.sum(rem & orig & (T["rating"] == c))) for c in (1, 2, 3, 4, 5)})
    d["rm_inj"] = int(np.sum(rem & T["inj"]))
    return d


for fam, alphas in P["alphas"].items():
    for alpha in alphas:
        elig, excl = X.eligibility(st, U, fam, alpha)
        targets = sorted(p for p in U.pids if U.half[p] == C.FAM_HALF[fam] and p not in excl)
        for r in REPS:
            for p in targets:
                T = X.attack(st, U, fam, r, p, alpha)
                one = (T["rating"] == 1); L1 = int(np.sum(one & ~T["inj"]))
                for rho in P["rhos"]:
                    b, clipped = C.budget(rho, T["k"], U.n[p])
                    res, rem, sc = X.run_methods(U, T, b, "attack", fam, r, "target")
                    share = b / (U.n[p] + T["k"])
                    for m, o in res.items():
                        row = dict(fam=fam, alpha=alpha, rho=rho, r=r, p=int(p), method=m, k=T["k"], b=b, clipped=clipped, budget_share=share,
                                   D=o["D"], score_dev=o["score_dev"], capture=o["capture"], L=o["L"], retained_low=o["retained_low"],
                                   L1=L1, boundary=o["ulp_near"], **removed_counts(T, rem[m]))
                        if fam in ("A2", "A3") and m in X.SCORE_METHODS:
                            row["auc_1star"] = C.auc(sc[m][one & T["inj"]], sc[m][one & ~T["inj"]])
                            row["exp_random_capture"] = C.expected_random_capture(b, T["k"], L1)
                        rows.append(row)
# FF: controls matched to the targets of A1 and A2 (A3 shares A2's targets, alphas and budget rule)
for fam in ("A1", "A2"):
    for alpha in P["alphas"][fam]:
        elig, excl = X.eligibility(st, U, fam, alpha)
        targets = sorted(p for p in U.pids if U.half[p] == C.FAM_HALF[fam] and p not in excl)
        controls = sorted({ff[f"{SPLIT}/{p}"]["control"] for p in targets if ff[f"{SPLIT}/{p}"]["control"] is not None})
        npairs = sum(ff[f"{SPLIT}/{p}"]["control"] is not None for p in targets)
        for c in controls:
            Tc = X.clean_target(st, U, c)
            for rho in P["rhos"]:
                b, clipped = C.ff_budget(rho, alpha, U.n[c])
                first = None
                for r in REPS:
                    res, rem, _ = X.run_methods(U, Tc, b, "ff", fam, r, "control")
                    if first is None:
                        first = rem
                    for m, o in res.items():
                        if m in X.SCORE_METHODS and r > 0:
                            assert (rem[m] == first[m]).all(), "FF score method not deterministic"
                            continue
                        ffrows.append(dict(fam=fam, alpha=alpha, rho=rho, r=r, control=int(c), method=m, b=b, clipped=clipped,
                                           weight=sum(ff[f"{SPLIT}/{p}"]["control"] == c for p in targets),
                                           D=o["D"], score_dev=o["score_dev"], L=o["L"], retained_low=o["retained_low"], **removed_counts(Tc, rem[m])))
df = pd.DataFrame(rows); fdf = pd.DataFrame(ffrows)
df.to_csv(ROWS, index=False, compression="gzip"); fdf.to_csv(ROWS.replace(".csv.gz", "_ff.csv.gz"), index=False, compression="gzip")


# ---------------------------------------------------------------- summaries (descriptive; MC uncertainty over replicates)
def summarise(g):
    """g: rows of one (fam, alpha, rho, method)."""
    rep = g.groupby("r")
    out = dict(D=C.t_interval(rep.D.mean()), score_dev=C.t_interval(rep.score_dev.mean()))
    capt = g.dropna(subset=["capture"]); out["capture"] = C.t_interval(capt.groupby("r").capture.mean())
    e = g[g.L >= 1]
    out["TRR"] = C.t_interval(e.groupby("r").retained_low.sum() / e.groupby("r").L.sum())
    out["TRR_unweighted"] = C.t_interval(e.groupby("r").apply(lambda x: (x.retained_low / x.L).mean(), include_groups=False))
    for c in (1, 2, 3, 4, 5):
        s = g.groupby("r")[[f"rm{c}", f"n{c}"]].sum()
        out[f"del_{c}"] = float((s[f"rm{c}"] / s[f"n{c}"]).mean()) if s[f"n{c}"].iloc[0] else None
    out["boundary"] = g.boundary.value_counts().to_dict()
    out["clipped_runs"] = int(g.clipped.sum()); out["budget_share_max"] = float(g.budget_share.max())
    out["n_targets"] = int(g.p.nunique()); out["n_targets_L_ge1"] = int(e.p.nunique())
    if "auc_1star" in g and g.auc_1star.notna().any():
        a = g.dropna(subset=["auc_1star"]); out["auc_1star"] = C.t_interval(a.groupby("r").auc_1star.mean()); out["auc_targets"] = int(a.p.nunique())
        s = g.groupby("r")[["rm_inj", "exp_random_capture"]].sum(); out["enrichment"] = C.t_interval(s.rm_inj / s.exp_random_capture)
    return out


summ = dict(label="EXPLORATORY - not confirmatory; fixed test cohort and attack generator; intervals = Monte Carlo uncertainty only",
            plan_record_sha256=stored, started_utc=plan["frozen_utc"], finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            model=st.ridge_meta, cells={}, D_rel={}, mechanism_contrast={}, ff={}, weight_concentration={})
for (fam, alpha, rho, m), g in df.groupby(["fam", "alpha", "rho", "method"]):
    if m == "LAF" and rho != 0.5:
        continue
    key = f"{fam}|{alpha}|{rho}|{m}"
    summ["cells"][key] = summarise(g)
    b0 = df[(df.fam == fam) & (df.alpha == alpha) & (df.rho == rho) & (df.method == "B0")]
    Dm = g.pivot(index="r", columns="p", values="D").values; Db = b0.pivot(index="r", columns="p", values="D").values
    th, se, _, _ = C.jackknife_ratio(Dm.sum(1), Db.sum(1))
    from scipy.stats import t as Tdist
    h = Tdist.ppf(0.975, len(REPS) - 1) * se
    summ["D_rel"][key] = dict(estimate=th, mcse_jackknife=se, lo=th - h, hi=th + h, cv_M=float(Db.sum(1).std(ddof=1) / Db.sum(1).mean()))
for alpha in sorted(set(P["alphas"]["A2"]) & set(P["alphas"]["A3"])):
    for rho in P["rhos"]:
        for m in X.ALL_METHODS:
            a2 = df[(df.fam == "A2") & (df.alpha == alpha) & (df.rho == rho) & (df.method == m)].set_index(["r", "p"]).capture
            a3 = df[(df.fam == "A3") & (df.alpha == alpha) & (df.rho == rho) & (df.method == m)].set_index(["r", "p"]).capture
            j = a2.index.intersection(a3.index)
            if len(j) and not (m == "LAF" and rho != 0.5):
                d = (a2.loc[j] - a3.loc[j]).groupby(level="r").mean()
                summ["mechanism_contrast"][f"{alpha}|{rho}|{m}"] = C.t_interval(d)
for (fam, alpha, rho, m), g in fdf.groupby(["fam", "alpha", "rho", "method"]):
    if m == "LAF":
        continue
    w = g.weight
    rep = g.groupby("r")
    e = g[g.L >= 1]
    cell = dict(D_weighted=C.t_interval(rep.apply(lambda x: np.average(x.D, weights=x.weight), include_groups=False)) if g.r.nunique() > 1 else
                dict(mean=float(np.average(g.D, weights=w)), mcse=0.0, lo=None, hi=None),
                score_dev_weighted=float(np.average(g.score_dev, weights=w)),
                TRR=float((e.retained_low * e.weight).sum() / (e.L * e.weight).sum()) if len(e) else None,
                del_1=float((g.rm1 * w).sum() / (g.n1 * w).sum()) if (g.n1 * w).sum() else None,
                controls=int(g.control.nunique()), pairs=int(g.drop_duplicates("control").weight.sum()), deterministic=m in X.SCORE_METHODS)
    summ["ff"][f"{fam}|{alpha}|{rho}|{m}"] = cell
for fam, alphas in P["alphas"].items():
    for alpha in alphas:
        g = df[(df.fam == fam) & (df.alpha == alpha) & (df.method == "B0") & (df.r == 0) & (df.rho == P["rhos"][0])]
        summ["weight_concentration"][f"{fam}|{alpha}"] = dict(value=C.weight_concentration(g.L), note="diagnostic only, not an effective sample size",
                                                               targets_L_ge1=int((g.L >= 1).sum()), L_max=int(g.L.max()))
summ["access"] = dict(universes_built=sorted(st.universes))
json.dump(summ, open(SUMMARY, "x"), indent=1, default=float)
print("rows", len(df), "ff rows", len(fdf), "cells", len(summ["cells"]))
