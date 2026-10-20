"""Corpus preparation.
Turkish: Trendyol reviews, up to 120 non-empty reviews per product (seed 7); products split 50/50 into a
'reference' half (used to fit the rating-expectation model) and an 'evaluation' half.
English: Fake Reviews Dataset (Salminen et al., 2022): original reviews (label 0) and computer-generated ones (label 1).
Usage: python prepare.py <data_dir> <work_dir>
"""
import sys, re, json
import numpy as np, pandas as pd
D, W = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(7)
t = pd.read_csv(f"{D}/trendyol-turkish-product-reviews/trendyol_reviews_20251212_000835.csv")
t["comment"] = t["comment"].astype(str).str.strip()
t = t[t.comment.str.len() >= 3].drop_duplicates(["product_url", "comment"])
t = t.sample(frac=1, random_state=7).groupby("product_url").head(120)
prods = t.product_url.unique(); rng.shuffle(prods)
half = set(prods[: len(prods) // 2])
t["part"] = np.where(t.product_url.isin(half), "reference", "evaluation")
slug = lambda u: re.sub(r"-p-\d+.*$", "", u.rstrip("/").split("/")[-1]).replace("-", " ")
t["product_name"] = t.product_url.map(slug)
t["pid"] = t.product_url.astype("category").cat.codes
t = t.reset_index(drop=True); t["rid"] = ["tr-%d" % i for i in range(len(t))]
t[["rid", "pid", "product_name", "category", "point", "comment", "part"]].rename(columns={"point": "rating", "comment": "text"}).to_parquet(f"{W}/tr_corpus.parquet")
print("TR", len(t), t.part.value_counts().to_dict(), "products", t.pid.nunique())
f = pd.read_parquet(f"{D}/Fake-Reviews-Dataset/data/train-00000-of-00001.parquet")
f = f[f.category.map(f.category.value_counts()) > 1000].copy()
f["text"] = f.text.astype(str).str.strip(); f = f.drop_duplicates("text").reset_index(drop=True)
f["rid"] = ["en-%d" % i for i in range(len(f))]
f[["rid", "category", "rating", "text", "label"]].to_parquet(f"{W}/en_corpus.parquet")
print("EN", len(f), f.label.value_counts().to_dict())
