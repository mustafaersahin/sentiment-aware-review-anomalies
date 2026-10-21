"""Decision-support experiment: manipulation of product rankings and recovery by review filtering.

Products of the Turkish evaluation part with >= 40 reviews are ranked by their Bayesian-average rating
(prior: global mean with weight 20). Attacks (per seed):
  promotion  20 random products from the lower half receive their 6 LLM-written 5-star reviews plus
             6 copies of genuine 1-2 star reviews of the same product re-labelled 5 stars (rating inflation)
  demotion   20 random products from the upper half receive 12 genuine negative reviews taken from other products
             of another category, rated 1 star (review bombing)
Each detector then removes the q highest-scoring reviews (q = number of injected reviews, i.e. an oracle budget,
and q = 5% of all reviews, a fixed budget) and the ranking is recomputed.
Metrics: Kendall tau to the clean ranking, mean absolute rank error of attacked products, NDCG@20 w.r.t. the clean
ranking, share of injected reviews removed, share of genuine reviews removed.
Usage: python ranking.py <work_dir> <out_json>
"""
import sys, json, numpy as np, pandas as pd
from scipy.stats import kendalltau
sys.argv = sys.argv[:3]
W, OUT = sys.argv[1], sys.argv[2]
import importlib.util
spec = importlib.util.spec_from_file_location("detect_mod", __file__.replace("ranking.py", "detect_lib.py"))
dm = importlib.util.module_from_spec(spec); spec.loader.exec_module(dm)
tr, fk, E, S = dm.tr, dm.fk, dm.E, dm.S

def bayes_rank(r, ctx, prods, prior, m=20):
    sc = {}
    for p in prods:
        x = r[ctx == p]; sc[p] = (x.sum() + m * prior) / (len(x) + m)
    order = sorted(prods, key=lambda p: -sc[p]); return {p: i for i, p in enumerate(order)}

def ndcg(rank, ref, k=20):
    rel = {p: max(0, k - ref[p]) for p in ref}
    top = sorted(rank, key=lambda p: rank[p])[:k]
    dcg = sum(rel[p] / np.log2(i + 2) for i, p in enumerate(top))
    ideal = sorted(rel.values(), reverse=True)[:k]
    return dcg / sum(v / np.log2(i + 2) for i, v in enumerate(ideal))

g = tr[tr.part == "evaluation"]; counts = g.pid.value_counts(); prods = counts[counts >= 40].index.values
g = g[g.pid.isin(prods)]; gi = g.index.values
prior = g.rating.mean()
res = {}
for seed in range(5):
    rng = np.random.default_rng(100 + seed)
    d = dict(text=g.text.values, emb=E["tr"][gi], sent=S["tr"][gi], rating=g.rating.values.astype(float),
             ctx=g.pid.values.copy(), cat=g.category.values, y=np.zeros(len(g), int))
    clean = bayes_rank(d["rating"], d["ctx"], prods, prior)
    order = sorted(prods, key=lambda p: clean[p]); n = len(order)
    promo = rng.choice(order[n // 2:], 20, replace=False); demo = rng.choice(order[: n // 2], 20, replace=False)
    add = {k: [] for k in d}
    for p in promo:
        f = fk[fk.pid == p]
        for j in f.index:
            add["text"].append(f.text[j]); add["emb"].append(E["trfake"][j]); add["sent"].append(S["trfake"][j])
            add["rating"].append(5.0); add["ctx"].append(p); add["cat"].append(d["cat"][d["ctx"] == p][0]); add["y"].append(1)
        neg = np.where((d["ctx"] == p) & (d["rating"] <= 2) & (d["y"] == 0))[0]
        for i in rng.permutation(neg)[:6]:
            for k in d: add[k].append(d[k][i] if k not in ("rating", "y") else (5.0 if k == "rating" else 1))
    for p in demo:
        cat = d["cat"][d["ctx"] == p][0]
        neg = np.where((d["cat"] != cat) & (d["rating"] <= 2) & (d["y"] == 0))[0]
        for i in rng.choice(neg, 12, replace=False):
            for k in d: add[k].append({"rating": 1.0, "y": 1, "ctx": p}.get(k, d[k][i]))
    a = {k: np.concatenate([d[k], np.array(add[k]) if k != "emb" else np.vstack(add[k])]) for k in d}
    attacked = bayes_rank(a["rating"], a["ctx"], prods, prior)
    sc = dm.scores(a, "tr")
    sc["No filtering"] = None
    nin = int(a["y"].sum())
    def evalrank(rank):
        tau = kendalltau([clean[p] for p in prods], [rank[p] for p in prods])[0]
        err = np.mean([abs(rank[p] - clean[p]) for p in list(promo) + list(demo)])
        return {"tau": tau, "rank_err": err, "ndcg20": ndcg(rank, clean)}
    for m, s in sc.items():
        for bud_name, q in [("oracle", nin), ("5pct", int(0.05 * len(a["y"])))]:
            if s is None:
                keep = np.ones(len(a["y"]), bool); rem = np.zeros(len(a["y"]), bool)
            else:
                rem = np.zeros(len(a["y"]), bool); rem[np.argsort(-s)[:q]] = True; keep = ~rem
            rank = bayes_rank(a["rating"][keep], a["ctx"][keep], prods, prior)
            r = evalrank(rank); r["injected_removed"] = rem[a["y"] == 1].mean(); r["genuine_removed"] = rem[a["y"] == 0].mean()
            for k, v in r.items(): res.setdefault(bud_name, {}).setdefault(m, {}).setdefault(k, []).append(float(v))
    print("seed", seed, "products", len(prods), "injected", nin, "attacked tau", round(evalrank(attacked)["tau"], 3), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
for b in res:
    print("\n== budget", b)
    for m, v in sorted(res[b].items(), key=lambda kv: np.mean(kv[1]["rank_err"])):
        print(f"  {m:30s} tau {np.mean(v['tau']):.3f} rank_err {np.mean(v['rank_err']):.1f} ndcg20 {np.mean(v['ndcg20']):.3f} inj_rm {np.mean(v['injected_removed']):.2f} gen_rm {np.mean(v['genuine_removed']):.3f}")
