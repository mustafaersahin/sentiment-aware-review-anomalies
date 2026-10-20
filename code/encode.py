"""Sentence embeddings (paraphrase-multilingual-MiniLM-L12-v2) and off-the-shelf sentiment scores.
Turkish sentiment: savasy/bert-base-turkish-sentiment-cased (P(positive)); English: nlptown/bert-base-multilingual-uncased-sentiment
(expected star rating). Usage: python encode.py <work_dir> emb|sent"""
import sys, os, numpy as np, pandas as pd, torch
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
W, WHAT = sys.argv[1], sys.argv[2]
tr = pd.read_parquet(f"{W}/tr_corpus.parquet"); fk = pd.read_parquet(f"{W}/tr_llm_fake.parquet"); en = pd.read_parquet(f"{W}/en_corpus.parquet")
sets = {"tr": tr.text.tolist(), "trfake": fk.text.tolist(), "en": en.text.tolist()}
if WHAT == "emb":
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
    for k, v in sets.items():
        e = m.encode(v, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
        np.save(f"{W}/emb_{k}.npy", e.astype(np.float32)); print(k, e.shape, flush=True)
else:
    from transformers import pipeline
    for k, v in sets.items():
        name = "nlptown/bert-base-multilingual-uncased-sentiment" if k == "en" else "savasy/bert-base-turkish-sentiment-cased"
        p = pipeline("text-classification", model=name, top_k=None, truncation=True, max_length=256, device=-1)
        out = []
        for i in range(0, len(v), 64):
            for r in p(v[i:i + 64], batch_size=64):
                d = {x["label"].lower(): x["score"] for x in r}
                out.append(sum(int(l[0]) * s for l, s in d.items()) if k == "en" else d.get("positive", 0.0))
        np.save(f"{W}/sent_{k}.npy", np.array(out, dtype=np.float32)); print(k, len(out), flush=True)
