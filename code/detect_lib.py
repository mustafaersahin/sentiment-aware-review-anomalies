"""Anomaly scenarios and detectors for sentiment-aware anomaly detection in product reviews.

Scenarios (anomalies are injected into genuine reviews of the evaluation part; 5 seeds):
  TR-flip      rating-text inconsistency: originally 1-2 star reviews re-labelled 5 stars and 5-star reviews re-labelled 1 star
  TR-llm       LLM-written promotional 5-star reviews added to their products
  TR-offtopic  genuine reviews moved to a product of another category (rating kept)
  TR-mixed     the three types together
  EN-cg        computer-generated reviews (Fake Reviews Dataset) among original ones
  EN-flip / EN-offtopic / EN-mixed  as above on the original English reviews (context = product category)
Detectors produce a score per review (higher = more anomalous); all are unsupervised with respect to the anomaly labels.
Usage: python detect.py <work_dir> <out_json>
"""
import sys, re, json, numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import rankdata

W = sys.argv[1]
CONT = 0.05
tr = pd.read_parquet(f"{W}/tr_corpus.parquet"); fk = pd.read_parquet(f"{W}/tr_llm_fake.parquet"); en = pd.read_parquet(f"{W}/en_corpus.parquet")
E = {"tr": np.load(f"{W}/emb_tr.npy"), "trfake": np.load(f"{W}/emb_trfake.npy"), "en": np.load(f"{W}/emb_en.npy")}
S = {"tr": np.load(f"{W}/sent_tr.npy"), "trfake": np.load(f"{W}/sent_trfake.npy"), "en": np.load(f"{W}/sent_en.npy")}

# ---- rating-expectation models fitted on reference data only (weak supervision from the ratings themselves)
ref = (tr.part == "reference").values
rid_tr = Ridge(alpha=10.0).fit(E["tr"][ref], tr.rating.values[ref])
en_ref = np.random.default_rng(1).random(len(en)) < 0.5
en_ok = en.label.values == 0
rid_en = Ridge(alpha=10.0).fit(E["en"][en_ref & en_ok], en.rating.values[en_ref & en_ok])

def build(lang, kind, seed):
    """Returns a dict of arrays for one test set: text, emb, sent, rating, context id, label."""
    rng = np.random.default_rng(seed)
    if lang == "tr":
        g = tr[tr.part == "evaluation"].copy(); gi = g.index.values
        base = dict(text=g.text.values, emb=E["tr"][gi], sent=S["tr"][gi], rating=g.rating.values.astype(float),
                    ctx=g.pid.values.copy(), cat=g.category.values, y=np.zeros(len(g), int))
        n = int(CONT * len(g))
    else:
        g = en[(~en_ref) & (en.label == 0)].copy(); gi = g.index.values
        base = dict(text=g.text.values, emb=E["en"][gi], sent=S["en"][gi], rating=g.rating.values.astype(float),
                    ctx=g.category.astype("category").cat.codes.values.copy(), cat=g.category.values, y=np.zeros(len(g), int))
        n = int(CONT * len(g))
    d = {k: v.copy() for k, v in base.items()}
    kinds = ["flip", "llm" if lang == "tr" else "cg", "offtopic"] if kind == "mixed" else [kind]
    per = n // len(kinds)
    for k in kinds:
        if k == "flip":
            free = np.where(d["y"] == 0)[0]
            low = rng.permutation(free[d["rating"][free] <= 2])[: per // 2]
            high = rng.permutation(free[d["rating"][free] == 5])[: per - len(low)]
            d["rating"][low] = 5.0; d["rating"][high] = 1.0; d["y"][low] = 1; d["y"][high] = 1
        elif k == "offtopic":
            free = rng.permutation(np.where(d["y"] == 0)[0])[: per]
            for i in free:
                others = np.where(d["cat"] != d["cat"][i])[0]
                d["ctx"][i] = d["ctx"][rng.choice(others)]
            d["y"][free] = 1
        elif k in ("llm", "cg"):
            if k == "llm":
                take = rng.permutation(len(fk))[: per]
                add = dict(text=fk.text.values[take], emb=E["trfake"][take], sent=S["trfake"][take],
                           rating=np.full(len(take), 5.0), ctx=fk.pid.values[take],
                           cat=np.array([tr.category.values[tr.pid.values == p][0] for p in fk.pid.values[take]]), y=np.ones(len(take), int))
            else:
                pool = en[(~en_ref) & (en.label == 1)]
                take = rng.permutation(len(pool))[: per]; pi = pool.index.values[take]
                cmap = dict(zip(g.category.astype("category").cat.categories, range(100)))
                add = dict(text=pool.text.values[take], emb=E["en"][pi], sent=S["en"][pi], rating=pool.rating.values[take].astype(float),
                           ctx=np.array([cmap.get(c, 0) for c in pool.category.values[take]]), cat=pool.category.values[take], y=np.ones(len(take), int))
            d = {kk: np.concatenate([d[kk], add[kk]]) for kk in d}
    return d

def rz(x):
    """robust z-score (median / MAD)"""
    m = np.median(x); s = np.median(np.abs(x - m)) * 1.4826 + 1e-9
    return (x - m) / s

def style_features(texts):
    """surface style of a review: length, word length, capitalisation, punctuation and symbol use"""
    f = []
    for t in texts:
        w = t.split(); n = max(1, len(w))
        f.append([len(w), np.mean([len(x) for x in w]) if w else 0, t[:1].isupper(), t.rstrip()[-1:] in ".!?",
                  sum(c in ",;:" for c in t) / n, sum(c.isupper() for c in t) / max(1, len(t)),
                  len(re.findall(r"[^\w\s,.!?;:'\"-]", t)) / n, t.count(". ") / n])
    return np.array(f, float)

def zr(x):
    return (rankdata(x) - 0.5) / len(x)          # rank-normalise to (0,1)

def scores(d, lang):
    X = d["emb"]; r = d["rating"]; out = {}
    rng = np.random.default_rng(0)
    # text-only global detectors on embeddings
    out["IF (text)"] = -IsolationForest(n_estimators=200, random_state=0).fit(X).score_samples(X)
    out["LOF (text)"] = -LocalOutlierFactor(n_neighbors=20).fit(X).negative_outlier_factor_
    fit = rng.choice(len(X), min(4000, len(X)), replace=False)
    out["OCSVM (text)"] = -OneClassSVM(nu=0.05, gamma="scale").fit(X[fit]).score_samples(X)
    ae = MLPRegressor(hidden_layer_sizes=(64, 16, 64), max_iter=60, random_state=0).fit(X[fit], X[fit])
    out["AE (text)"] = ((ae.predict(X) - X) ** 2).sum(1)
    tf = TfidfVectorizer(min_df=2, max_features=30000, ngram_range=(1, 2)).fit_transform(d["text"])
    Xt = TruncatedSVD(100, random_state=0).fit_transform(tf)
    out["IF (TF-IDF)"] = -IsolationForest(n_estimators=200, random_state=0).fit(Xt).score_samples(Xt)
    # naive joint: rating appended to the embedding
    Xr = np.hstack([X, (r[:, None] - 3) / 2])
    out["IF (text+rating)"] = -IsolationForest(n_estimators=200, random_state=0).fit(Xr).score_samples(Xr)
    # sentiment consistency
    if lang == "tr":
        out["Sentiment gap (pretrained)"] = np.abs((r - 1) / 4 - d["sent"])
        exp = rid_tr.predict(X)
    else:
        out["Sentiment gap (pretrained)"] = np.abs(r - d["sent"])
        exp = rid_en.predict(X)
    out["Sentiment gap (rating model)"] = np.abs(r - exp)
    # context: dissimilarity to the review's product (TR) or category (EN) and redundancy within it
    ctx_dist = np.zeros(len(X)); redund = np.zeros(len(X))
    for c in np.unique(d["ctx"]):
        idx = np.where(d["ctx"] == c)[0]
        if len(idx) < 3: continue
        sims = X[idx] @ X[idx].T; np.fill_diagonal(sims, np.nan)
        cen = X[idx].sum(0)
        loo = (cen[None, :] - X[idx]) / (len(idx) - 1)            # leave-one-out centroid
        ctx_dist[idx] = 1 - (X[idx] * loo).sum(1) / (np.linalg.norm(loo, axis=1) + 1e-9)
        k = min(5, len(idx) - 1)
        redund[idx] = np.sort(np.nan_to_num(sims, nan=-1), axis=1)[:, -k:].mean(1)
    out["Context distance"] = ctx_dist
    out["Redundancy"] = redund
    Fs = style_features(d["text"])
    out["Style (IF)"] = -IsolationForest(n_estimators=200, random_state=0).fit(Fs).score_samples(Fs)
    # proposed SAAD: maximum robust z-score of the relational signals
    sent = (rz(out["Sentiment gap (pretrained)"]) + rz(out["Sentiment gap (rating model)"])) / 2
    out["Sentiment gap (ensemble)"] = sent
    ctx, sty = rz(out["Context distance"]), rz(out["Style (IF)"])
    out["SAAD (proposed)"] = np.max([sent, ctx, sty], axis=0)
    out["SAAD w/o style"] = np.max([sent, ctx], axis=0)
    out["SAAD w/o context"] = np.max([sent, sty], axis=0)
    out["SAAD w/o sentiment"] = np.max([ctx, sty], axis=0)
    out["SAAD, rank fusion"] = np.max([zr(sent), zr(ctx), zr(sty)], axis=0)
    out["SAAD, sum fusion"] = sent + ctx + sty
    return out

