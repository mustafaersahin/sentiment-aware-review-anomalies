"""Re-runs, from the saved inputs, the analyses of the earlier audit turns that were only computed in memory,
and compares them with the values quoted in those turns. Uses the ORIGINAL experiment code (detect_lib.py,
ranking.py logic); nothing in sad/work or sad/paper is written. Usage: python diag_prior.py <sad_dir> <out_json>"""
import sys, json, importlib.util
import numpy as np, pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import average_precision_score as ap, cohen_kappa_score

SAD, OUT = sys.argv[1], sys.argv[2]
W = f"{SAD}/work"
sys.argv = ["detect_lib", W]
spec = importlib.util.spec_from_file_location("dl", f"{SAD}/code/detect_lib.py"); dm = importlib.util.module_from_spec(spec); spec.loader.exec_module(dm)
tr, fk, E, S = dm.tr, dm.fk, dm.E, dm.S
J = json.load(open(f"{W}/ranking.json"))
res, prev = {}, {}

def bayes_rank(r, ctx, prods, prior, m=20):
    sc = {p: (r[ctx == p].sum() + m * prior) / ((ctx == p).sum() + m) for p in prods}
    order = sorted(prods, key=lambda p: -sc[p]); return {p: i for i, p in enumerate(order)}
def ndcg(rank, ref, k=20):
    rel = {p: max(0, k - ref[p]) for p in ref}; top = sorted(rank, key=lambda p: rank[p])[:k]
    dcg = sum(rel[p] / np.log2(i + 2) for i, p in enumerate(top)); ideal = sorted(rel.values(), reverse=True)[:k]
    return dcg / sum(v / np.log2(i + 2) for i, v in enumerate(ideal))

g = tr[tr.part == "evaluation"]; counts = g.pid.value_counts(); prods = counts[counts >= 40].index.values
g = g[g.pid.isin(prods)]; gi = g.index.values; prior = g.rating.mean(); r0 = g.rating.values.astype(float)

# (1) ridge predictions and |r - pred| per star (no attack)
pred = dm.rid_tr.predict(E["tr"][gi]); gap = np.abs(r0 - pred)
res["pool"] = dict(products=len(prods), reviews=len(g), star_counts={int(k): int(v) for k, v in g.rating.value_counts().sort_index().items()})
res["pred_by_star"] = {int(c): dict(n=int((r0 == c).sum()), pred_mean=float(pred[r0 == c].mean()), gap_p10=float(np.percentile(gap[r0 == c], 10)),
                                    gap_median=float(np.median(gap[r0 == c])), gap_p90=float(np.percentile(gap[r0 == c], 90))) for c in (1, 2, 3, 4, 5)}
prev["pred_by_star_mean"] = {1: 3.77, 2: 3.89, 3: 4.23, 4: 4.62, 5: 4.78}

# (2) no-attack removal profile
d = dict(text=g.text.values, emb=E["tr"][gi], sent=S["tr"][gi], rating=r0.copy(), ctx=g.pid.values.copy(), cat=g.category.values, y=np.zeros(len(g), int))
sc0 = dm.scores(d, "tr")
for q in (int(0.05 * len(g)), 1136):
    for m in ["Sentiment gap (rating model)", "Sentiment gap (ensemble)", "SAAD w/o style", "SAAD (proposed)", "IF (text)"]:
        rem = np.zeros(len(g), bool); rem[np.argsort(-sc0[m])[:q]] = True
        res.setdefault("no_attack", {}).setdefault(str(q), {})[m] = {"removed_by_star": {int(c): [int(rem[r0 == c].sum()), int((r0 == c).sum())] for c in (1, 2, 3, 4, 5)},
                                                                     "low_removed": [int(rem[r0 <= 2].sum()), int((r0 <= 2).sum())]}
prev["no_attack_rm_q1114"] = {1: [419, 419], 2: [173, 201], 3: [384, 727], 4: [1, 2442], 5: [137, 18504]}
prev["no_attack_rm_low_q1136"] = [593, 620]

# (3) ranking experiment reproduction + per-type profile + baselines + decomposition + promo/demo split
METHODS = ["No filtering", "IF (text)", "OCSVM (text)", "AE (text)", "Sentiment gap (pretrained)", "Sentiment gap (rating model)", "Sentiment gap (ensemble)", "Context distance", "SAAD w/o style", "SAAD (proposed)"]
agg, maxdiff, seeds = {}, 0.0, []
for seed in range(5):
    rng = np.random.default_rng(100 + seed)
    d = dict(text=g.text.values, emb=E["tr"][gi], sent=S["tr"][gi], rating=r0.copy(), ctx=g.pid.values.copy(), cat=g.category.values, y=np.zeros(len(g), int))
    clean = bayes_rank(d["rating"], d["ctx"], prods, prior); order = sorted(prods, key=lambda p: clean[p]); n = len(order)
    promo = rng.choice(order[n // 2:], 20, replace=False); demo = rng.choice(order[: n // 2], 20, replace=False)
    add = {k: [] for k in d}; typ = []
    for p in promo:
        f = fk[fk.pid == p]
        for j in f.index:
            add["text"].append(f.text[j]); add["emb"].append(E["trfake"][j]); add["sent"].append(S["trfake"][j])
            add["rating"].append(5.0); add["ctx"].append(p); add["cat"].append(d["cat"][d["ctx"] == p][0]); add["y"].append(1); typ.append("llm")
        neg = np.where((d["ctx"] == p) & (d["rating"] <= 2) & (d["y"] == 0))[0]
        for i in rng.permutation(neg)[:6]:
            for k in d: add[k].append(d[k][i] if k not in ("rating", "y") else (5.0 if k == "rating" else 1))
            typ.append("flipcopy")
    for p in demo:
        cat = d["cat"][d["ctx"] == p][0]; neg = np.where((d["cat"] != cat) & (d["rating"] <= 2) & (d["y"] == 0))[0]
        for i in rng.choice(neg, 12, replace=False):
            for k in d: add[k].append({"rating": 1.0, "y": 1, "ctx": p}.get(k, d[k][i]))
            typ.append("bomb")
    a = {k: np.concatenate([d[k], np.array(add[k]) if k != "emb" else np.vstack(add[k])]) for k in d}
    T = np.array(["orig"] * len(g) + typ); N = len(T); q = int(0.05 * N); ostar = np.concatenate([r0, np.zeros(len(typ))])
    seeds.append(dict(N=N, q=q, injected=int((T != "orig").sum()), llm=int((T == "llm").sum()), flipcopy=int((T == "flipcopy").sum()), bomb=int((T == "bomb").sum()), one_star_after=int((a["rating"] == 1).sum())))
    sc = dm.scores(a, "tr"); br = np.random.default_rng(1000 + seed)
    def ev(rem, sub=None):
        keep = ~rem; rank = bayes_rank(a["rating"][keep], a["ctx"][keep], prods, prior)
        o = {"tau": kendalltau([clean[p] for p in prods], [rank[p] for p in prods])[0], "rank_err": np.mean([abs(rank[p] - clean[p]) for p in list(promo) + list(demo)]),
             "ndcg20": ndcg(rank, clean), "n_removed": int(rem.sum()), "injected_removed": rem[a["y"] == 1].mean(), "genuine_removed": rem[a["y"] == 0].mean(),
             "err_promo": np.mean([abs(rank[p] - clean[p]) for p in promo]), "err_demo": np.mean([abs(rank[p] - clean[p]) for p in demo])}
        for t in ["llm", "flipcopy", "bomb"]: o["rm_" + t] = rem[T == t].mean()
        for c in (1, 2, 3, 4, 5): o[f"orig{c}_rm"] = rem[(T == "orig") & (ostar == c)].mean()
        ro = rem & (T == "orig"); o["removed_orig_low_share"] = (ostar[ro] <= 2).mean() if ro.any() else np.nan
        return o
    R = {}
    for m in METHODS:
        rem = np.zeros(N, bool)
        if m != "No filtering": rem[np.argsort(-sc[m])[:q]] = True
        R[m] = ev(rem)
        for k in ["tau", "rank_err", "ndcg20", "injected_removed", "genuine_removed"]: maxdiff = max(maxdiff, abs(R[m][k] - J["5pct"][m][k][seed]))
    ones = np.where(a["rating"] == 1)[0]
    rem = np.zeros(N, bool); rem[br.permutation(N)[:q]] = True; R["Random"] = ev(rem)
    rem = np.zeros(N, bool); rem[br.permutation(ones)[:q]] = True; R["Random 1-star (strict)"] = ev(rem)
    rem = np.zeros(N, bool); rem[ones[:q]] = True
    if len(ones) < q: rest = np.setdiff1d(np.arange(N), ones); rem[br.permutation(rest)[: q - len(ones)]] = True
    R["Random 1-star + random fill"] = ev(rem)
    lo = np.lexsort((br.random(N), a["rating"])); rem = np.zeros(N, bool); rem[lo[:q]] = True; R["Lowest rating first"] = ev(rem)
    # decomposition of the rating-model gap removal
    gm = np.zeros(N, bool); gm[np.argsort(-sc["Sentiment gap (rating model)"])[:q]] = True; inj = T != "orig"
    for name, rm in [("decomp: only injected it caught", gm & inj), ("decomp: only originals it caught", gm & ~inj), ("decomp: all injected", inj),
                     ("decomp: bombs only", T == "bomb"), ("decomp: flip+llm only", (T == "llm") | (T == "flipcopy"))]:
        R[name] = ev(rm)
    for m, o in R.items(): agg.setdefault(m, []).append(o)
res["ranking_seeds"] = seeds
res["ranking_maxdiff_vs_json"] = maxdiff
res["ranking"] = {m: {k: float(np.nanmean([o[k] for o in L])) for k in L[0]} | {"rank_err_per_seed": [float(o["rank_err"]) for o in L]} for m, L in agg.items()}
prev["ranking"] = {"Sentiment gap (rating model)": dict(rank_err=11.5, rm_llm=0.0, rm_flipcopy=0.278, rm_bomb=0.995, orig1_rm=0.995, orig2_rm=0.736, orig3_rm=0.361, removed_orig_low_share=0.644, err_promo=12.9, err_demo=10.1),
                   "No filtering": dict(rank_err=56.1, err_promo=16.0, err_demo=96.2),
                   "Random 1-star (strict)": dict(rank_err=9.7, n_removed=659), "Random 1-star + random fill": dict(rank_err=10.2),
                   "Lowest rating first": dict(rank_err=14.7), "Random": dict(rank_err=56.6),
                   "decomp: only injected it caught": dict(rank_err=2.1), "decomp: only originals it caught": dict(rank_err=73.0),
                   "decomp: all injected": dict(rank_err=0.0), "decomp: bombs only": dict(rank_err=1.6), "decomp: flip+llm only": dict(rank_err=53.7)}
prev["ranking_seeds_injected"] = [437, 429, 431, 431, 432]

# (4) flip detection vs lowest-rating baseline
for lang in ["tr", "en"]:
    rows = []
    for seed in range(5):
        d = dm.build(lang, "flip", seed); y = d["y"]; r = d["rating"]
        rm = np.abs(r - (dm.rid_tr if lang == "tr" else dm.rid_en).predict(d["emb"]))
        pre = np.abs((r - 1) / 4 - d["sent"]) if lang == "tr" else np.abs(r - d["sent"]); ens = (dm.rz(pre) + dm.rz(rm)) / 2
        low = -r + 1e-6 * np.random.default_rng(seed).random(len(r))
        infl = (y == 1) & (r == 5); defl = (y == 1) & (r == 1); sub = lambda s, m: ap(y[m | (y == 0)], s[m | (y == 0)])
        rows.append([ap(y, rm), ap(y, ens), ap(y, low), sub(rm, infl), sub(rm, defl), sub(ens, infl), sub(ens, defl), sub(low, defl)])
    res[f"flip_{lang}"] = dict(zip(["rm", "ens", "lowest", "rm_infl", "rm_defl", "ens_infl", "ens_defl", "lowest_defl"], np.mean(rows, 0).tolist()))
prev["flip_tr"] = dict(rm=0.755, ens=0.862, lowest=0.493, rm_infl=0.231, rm_defl=0.993, ens_infl=0.530, ens_defl=0.993, lowest_defl=0.913)
prev["flip_en"] = dict(rm=0.650, ens=0.907, lowest=0.219, rm_infl=0.131, rm_defl=0.923, ens_infl=0.701, ens_defl=0.966, lowest_defl=0.383)

# (5) natural audit: composition and agreement
k = pd.read_csv(f"{W}/natural_audit.csv").merge(pd.read_csv(f"{W}/human_audit.csv"), on="rid")
res["audit_crosstab"] = {f"{s}/{int(r)}": {lab: int(v) for lab, v in g2.human.value_counts().items()} for (s, r), g2 in k.groupby(["sample", "rating"])}
b = k[(k.human != "unclear") & (k.label != "unclear")]
res["audit"] = dict(top_inc=float((k[k["sample"] == "top200"].human == "inconsistent").mean()), random_inc=float((k[k["sample"] == "random200"].human == "inconsistent").mean()),
                    kappa=float(cohen_kappa_score(b.human, b.label)), n_bin=len(b))
prev["audit"] = dict(top_inc=0.27, random_inc=0.04, kappa=0.79, n_bin=370, top200_1star_inconsistent=[31, 144])
json.dump(dict(reproduced=res, previously_quoted=prev), open(OUT, "w"), indent=1, default=float)
print("done; maxdiff", maxdiff)
