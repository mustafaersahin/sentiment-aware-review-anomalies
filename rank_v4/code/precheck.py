"""Pre-checks for protocol v4.2 (read-only on the experiment inputs; writes only to rank_v4/out).
Usage: python precheck.py <sad_dir> <out_dir>"""
import sys, json, hashlib, subprocess, platform, math, re
import numpy as np, pandas as pd, sklearn, scipy
import common as C, exp as X

SAD, OUT = sys.argv[1], sys.argv[2]
W = f"{SAD}/work"
rep = {}

# 1. hashes, code commit, environment
files = ["code/ranking.py", "code/detect_lib.py", "code/prepare.py", "code/gen_fake_tr.py", "code/encode.py",
         "work/tr_corpus.parquet", "work/emb_tr.npy", "work/sent_tr.npy", "work/tr_llm_fake.parquet", "work/ranking.json",
         "work/detect.json", "work/human_audit.csv", "work/natural_audit.csv", "paper/main.tex", "paper/tables/ranking.tex",
         "data/trendyol-turkish-product-reviews/trendyol_reviews_20251212_000835.csv",
         "rank_v4/code/common.py", "rank_v4/code/exp.py", "rank_v4/PROTOCOL_v4.2.md"]
rep["sha256"] = {f: hashlib.sha256(open(f"{SAD}/{f}", "rb").read()).hexdigest() for f in files}
repo = f"{SAD}/../repo_sad"
rep["repo_commit"] = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
rep["repo_dirty_files"] = len(subprocess.run(["git", "-C", repo, "status", "--short"], capture_output=True, text=True).stdout.splitlines())
rep["env"] = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                  scipy=scipy.__version__, machine=platform.machine(), entropy=C.ENTROPY)

# 2. schema
tr = pd.read_parquet(f"{W}/tr_corpus.parquet"); E = np.load(f"{W}/emb_tr.npy"); S = np.load(f"{W}/sent_tr.npy")
sch = dict(columns=list(tr.columns), dtypes={c: str(t) for c, t in tr.dtypes.items()}, n_rows=len(tr),
           rid_unique=bool(tr.rid.is_unique), rid_format_ok=bool(tr.rid.str.fullmatch(r"tr-\d+").all()),
           rid_has_pipe=bool(tr.rid.str.contains(r"\|").any()),
           ratings=sorted(tr.rating.unique().tolist()), text_null=int(tr.text.isna().sum()), parts=tr.part.value_counts().to_dict(),
           emb_shape=list(E.shape), emb_finite=bool(np.isfinite(E).all()), sent_shape=list(S.shape),
           sent_range=[float(S.min()), float(S.max())], sent_finite=bool(np.isfinite(S).all()),
           duplicate_norm_text_within_product=int(tr.assign(n=[C.norm_text(t) for t in tr.text]).duplicated(["pid", "n"]).sum()))
rep["schema"] = sch
raw = pd.read_csv(f"{SAD}/data/trendyol-turkish-product-reviews/trendyol_reviews_20251212_000835.csv", usecols=["product_url", "category"])
rep["raw_csv"] = dict(rows=len(raw), unique_product_urls=int(raw.product_url.nunique()),
                      products_per_category=raw.drop_duplicates("product_url").category.value_counts().to_dict())

# 3. setup: splits, source products, universes
st = X.Setup(W)
ev = tr[tr.part == "evaluation"]
rep["counts"] = dict(eval_products=int(ev.pid.nunique()), ranking_pool=len(st.pool), val=len(st.split_pids["val"]),
                     test=len(st.split_pids["test"]), reference_products=int(tr[tr.part == "reference"].pid.nunique()),
                     source_products=len(st.src_pids), source_reviews=len(st.src_idx),
                     ridge_fit_reviews=int(((tr.part == "reference").values & ~np.isin(st.pid, st.src_pids)).sum()))
dist = {}
for s, U in st.universes.items():
    for half in ("lower", "upper"):
        ps = [p for p in U.pids if U.half[p] == half]
        n = np.array([U.n[p] for p in ps]); L = np.array([U.L[p] for p in ps])
        dist[f"{s}/{half}"] = dict(products=len(ps), n_min=int(n.min()), n_median=float(np.median(n)), n_max=int(n.max()),
                                   L_min=int(L.min()), L_median=float(np.median(L)), L_max=int(L.max()),
                                   L_ge1=int((L >= 1).sum()), L_class_counts=[int(sum(C.l_class(x) == i for x in L)) for i in range(4)],
                                   n_reviews=int(n.sum()))
    rep.setdefault("universe", {})[s] = dict(mu=U.mu, n_reviews=int(sum(U.n.values())))
rep["n_L_distribution"] = dist
# ridge refit: mean prediction per star on the ranking pool (compare with the original ridge diagnostics)
pool_idx = np.where(np.isin(st.pid, st.pool))[0]
rep["ridge_refit_pred_by_star"] = {int(c): float(st.pred[pool_idx][st.rating[pool_idx] == c].mean()) for c in (1, 2, 3, 4, 5)}
src_r = st.rating[st.src_idx]
rep["source_pool_by_cat_star"] = {cat: {int(c): int(((st.cat[st.src_idx] == cat) & (src_r == c)).sum()) for c in (1, 2, 3, 4, 5)}
                                  for cat in sorted(set(st.cat[st.src_idx]))}

# 4. pool adequacy per target / family / alpha
ade, excl = [], {}
for s, U in st.universes.items():
    for fam in ("A1", "A2", "A3"):
        for p in [q for q in U.pids if U.half[q] == C.FAM_HALF[fam]]:
            pool = st.source_pool(fam, p, set(st.norm[U.idx[p]]))
            for a in C.ALPHAS:
                k = math.ceil(a * U.n[p]); ade.append(dict(split=s, fam=fam, p=int(p), alpha=a, k=k, pool=len(pool), ok=len(pool) >= k))
ade = pd.DataFrame(ade)
# A2 exclusion propagates to A3 (paired targets)
a2bad = set(map(tuple, ade[(ade.fam == "A2") & ~ade.ok][["split", "p", "alpha"]].values.tolist()))
ade["ok_paired"] = [ok and not (f == "A3" and (s, p, a) in a2bad) for ok, f, s, p, a in ade[["ok", "fam", "split", "p", "alpha"]].values]
for (s, f, a), g in ade.groupby(["split", "fam", "alpha"]):
    excl[f"{s}/{f}/{a}"] = dict(targets=len(g), excluded_pool=int((~g.ok).sum()), excluded_paired=int((~g.ok_paired).sum()),
                                share_excluded=float((~g.ok_paired).mean()), min_pool=int(g.pool.min()), max_k=int(g.k.max()))
rep["pool_adequacy"] = excl
feas = {}
for f in ("A1", "A2", "A3"):
    feas[f] = [a for a in C.ALPHAS if all(excl[f"{s}/{f}/{a}"]["share_excluded"] <= 0.10 for s in ("val", "test"))]
rep["feasible_alphas"] = feas
# eligible targets for the K outcome (L_p >= 1) after exclusion
elig = {}
for (s, f, a), g in ade[ade.ok_paired].groupby(["split", "fam", "alpha"]):
    U = st.universes[s]; elig[f"{s}/{f}/{a}"] = int(sum(U.L[p] >= 1 for p in g.p))
rep["eligible_targets_L_ge1"] = elig

# 5. FF matching (pre-attack covariates only)
ff, quality = {}, {}
for s, U in st.universes.items():
    for half in ("lower", "upper"):
        ps = sorted(p for p in U.pids if U.half[p] == half)
        for p in ps:
            cand = [c for c in ps if c != p and C.l_class(U.L[c]) == C.l_class(U.L[p])]
            chosen, cal = None, None
            for caliper in (10, 20):
                cc = [c for c in cand if abs(U.n[c] - U.n[p]) <= caliper]
                if cc:
                    best = min(abs(U.n[c] - U.n[p]) for c in cc); cc = [c for c in cc if abs(U.n[c] - U.n[p]) == best]
                    same = [c for c in cc if st.cat[U.idx[c][0]] == st.cat[U.idx[p][0]]]
                    cc = sorted(same or cc)
                    chosen = int(cc[C.rng(5, p).integers(len(cc))]) if len(cc) > 1 else int(cc[0]); cal = caliper; break
            ff[f"{s}/{p}"] = dict(split=s, half=half, target=int(p), control=chosen, caliper=cal)
        rows = [v for v in ff.values() if v["split"] == s and v["half"] == half]
        m = [v for v in rows if v["control"] is not None]
        nt = np.array([U.n[v["target"]] for v in m]); nc = np.array([U.n[v["control"]] for v in m])
        lt = np.array([U.L[v["target"]] for v in m]); lc = np.array([U.L[v["control"]] for v in m])
        smd = lambda a, b: float((a.mean() - b.mean()) / math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)) if (a.var() + b.var()) > 0 else 0.0
        reuse = pd.Series([v["control"] for v in m]).value_counts()
        quality[f"{s}/{half}"] = dict(targets=len(rows), matched=len(m), caliper10=sum(v["caliper"] == 10 for v in m),
                                      caliper20=sum(v["caliper"] == 20 for v in m), unmatched=len(rows) - len(m),
                                      smd_n=smd(nt, nc), smd_L=smd(lt, lc), abs_n_diff_max=int(np.abs(nt - nc).max()) if m else None,
                                      max_reuse=int(reuse.max()) if m else None, distinct_controls=int(len(reuse)),
                                      b_diff_rho1_alpha02_max=int(max(abs(math.ceil(math.ceil(0.2 * a)) - math.ceil(math.ceil(0.2 * b))) for a, b in zip(nt, nc))) if m else None)
rep["ff_matching_quality"] = quality
json.dump(ff, open(f"{OUT}/ff_matches.json", "w"), indent=1)
ade.to_csv(f"{OUT}/pool_adequacy.csv", index=False)
json.dump(rep, open(f"{OUT}/precheck.json", "w"), indent=1, default=str)
print(json.dumps({k: rep[k] for k in ["counts", "n_L_distribution", "pool_adequacy", "feasible_alphas", "eligible_targets_L_ge1", "ff_matching_quality", "ridge_refit_pred_by_star", "schema", "raw_csv"]}, indent=1, default=str))
