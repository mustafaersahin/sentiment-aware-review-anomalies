"""Builds the blind source-polarity sample (protocol 14.7, as amended on 2026-10-21).
Per family (A1: 1-2 star sources, expected 'olumsuz'; A2: 5-star sources, expected 'olumlu') 100 source texts,
allocated to categories in proportion to the planned attack draws (sum of n_p over the family's target products in
validation and test; k_p = ceil(alpha*n_p) is proportional to n_p for every alpha), drawn in a seeded order within
each category, unique by normalised text within and across families. Writes:
  <sheet_csv>  opaque id + text only (for the annotator)
  <key_csv>    id -> rid, family, expected polarity, category, rating (NOT for the annotator)
  <report>     allocation, uniqueness checks and hashes (no labels)
Usage: python make_polarity_sheet.py <sad_dir> <sheet_csv> <key_csv> <report_json>"""
import sys, json, hashlib, math
import numpy as np, pandas as pd
import common as C, exp as X

SAD, SHEET, KEY, REP = sys.argv[1:5]
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
st = X.Setup(f"{SAD}/work")
cats = sorted(set(st.cat))
rep = dict(rule="protocol 14.7 (amended 2026-10-21): category quotas proportional to sum n_p of the family's targets (val+test)")
seen, rows = set(), []
for fam in ("A1", "A2"):
    T = [(s, p) for s, U in st.universes.items() for p in U.pids if U.half[p] == C.FAM_HALF[fam]]
    cat_of = lambda p: st.cat[st.pid == p][0]
    w = {c: sum(st.universes[s].n[p] for s, p in T if cat_of(p) == c) for c in cats}
    quotas = C.largest_remainder(w, C.POLARITY_N)
    alt = {str(a): C.largest_remainder({c: sum(math.ceil(a * st.universes[s].n[p]) for s, p in T if cat_of(p) == c) for c in cats}, C.POLARITY_N)
           for a in C.ALPHA_CONFIRM}
    pool = set(X.polarity_pool_rids(st, fam))
    cand, poolsize = {}, {}
    for ci, c in enumerate(cats):
        i = st.src_idx[(st.cat[st.src_idx] == c) & np.isin(st.rid[st.src_idx], list(pool))]
        i = i[np.argsort(st.rid_num[i], kind="stable")]
        poolsize[c] = dict(reviews=int(len(i)), unique_norm_texts=int(len(set(st.norm[i]))))
        order = C.rng(9, C.FAM_NO[fam], ci).permutation(len(i))
        cand[c] = [(st.rid[i[j]], st.norm[i[j]]) for j in order]
    acc, rej, short = C.stratified_unique_sample(cand, quotas, seen)
    for c, rids in acc.items():
        for r in rids:
            k = int(np.where(st.rid == r)[0][0])
            rows.append(dict(rid=r, family=fam, expected=C.POLARITY_EXPECTED[fam], category=c, source_pid=int(st.pid[k]),
                             rating=int(st.rating[k]), text=st.tr.text.values[k], norm_sha=hashlib.sha256(st.norm[k].encode()).hexdigest()[:16]))
    rep[fam] = dict(target_products=len(T), weights_sum_n=w, quotas=quotas, quotas_if_alpha=alt, pool=poolsize,
                    accepted={c: len(v) for c, v in acc.items()}, total=sum(len(v) for v in acc.values()),
                    rejected_duplicates=rej, shortfall=short)
df = pd.DataFrame(rows)
perm = C.rng(9, 0).permutation(len(df)); df = df.iloc[perm].reset_index(drop=True)
df.insert(0, "id", [f"K-{i + 1:03d}" for i in range(len(df))])
rep["checks"] = dict(n=len(df), unique_rid=bool(df.rid.is_unique), unique_norm_text=bool(df.norm_sha.is_unique),
                     families=df.family.value_counts().to_dict(), all_from_source_products=bool(df.source_pid.isin(st.src_pids).all()),
                     ratings_by_family={f: sorted(g.rating.unique().tolist()) for f, g in df.groupby("family")})
df[["id", "text"]].to_csv(SHEET, index=False)
df[["id", "rid", "family", "expected", "category", "source_pid", "rating", "norm_sha"]].to_csv(KEY, index=False)
rep["hashes"] = dict(code={f: sha(f"{SAD}/rank_v4/code/{f}") for f in ("make_polarity_sheet.py", "common.py", "exp.py")},
                     data={f: sha(f"{SAD}/{f}") for f in ("work/tr_corpus.parquet", "work/emb_tr.npy", "work/sent_tr.npy")},
                     sheet_csv=sha(SHEET), key_csv=sha(KEY), protocol=sha(f"{SAD}/rank_v4/PROTOCOL_v4.2.md"),
                     source_products_sha256=hashlib.sha256(",".join(map(str, map(int, st.src_pids))).encode()).hexdigest(),
                     entropy=C.ENTROPY)
json.dump(rep, open(REP, "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: rep[k] for k in ("A1", "A2", "checks")}, indent=1, ensure_ascii=False))
