"""Records the frozen pre-test metadata (protocol section 14): data revisions and hashes, the source-product
list and its hash, ridge_full / ridge_srcout model metadata, code hashes and the environment.
It computes no experiment outcome. Usage: python freeze_meta.py <sad_dir> <out_json>"""
import sys, json, hashlib, subprocess, platform, glob, os
import numpy as np, pandas as pd, sklearn
from sklearn.linear_model import Ridge
import common as C, exp as X

SAD, OUT = sys.argv[1], sys.argv[2]
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()


def hf_cache(d):
    """Revision and etag recorded by huggingface_hub when the files were downloaded."""
    out = {}
    for m in sorted(glob.glob(f"{d}/.cache/huggingface/download/**/*.metadata", recursive=True)):
        rev, etag, _ = open(m).read().split("\n")[:3]
        out[os.path.relpath(m, f"{d}/.cache/huggingface/download")[: -len(".metadata")]] = dict(revision=rev, etag=etag)
    return out


meta = dict(entropy=C.ENTROPY)
meta["datasets"] = {
    "trendyol": dict(repo="Doukan/trendyol-turkish-product-reviews", download_cache=hf_cache(f"{SAD}/data/trendyol-turkish-product-reviews"),
                     hub_main_sha_checked_2026_10_21="3b390598dac913449a75b624a06e985a1c5d726c",
                     csv_sha256=sha(f"{SAD}/data/trendyol-turkish-product-reviews/trendyol_reviews_20251212_000835.csv"),
                     readme_sha256=sha(f"{SAD}/data/trendyol-turkish-product-reviews/README.md")),
    "fake_reviews": dict(repo="theArijitDas/Fake-Reviews-Dataset", download_cache=hf_cache(f"{SAD}/data/Fake-Reviews-Dataset"))}
meta["data_files_sha256"] = {f: sha(f"{SAD}/{f}") for f in ["work/tr_corpus.parquet", "work/emb_tr.npy", "work/sent_tr.npy"]}
st = X.Setup(f"{SAD}/work")
src = [int(p) for p in st.src_pids]
meta["source_products"] = dict(rule="30% of reference products, SeedSequence(ENTROPY, spawn_key=(1,)) permutation, first round(0.3*238)",
                               n=len(src), pids=src, sha256=hashlib.sha256(",".join(map(str, src)).encode()).hexdigest(),
                               reviews=int(len(st.src_idx)),
                               by_category=pd.Series([st.cat[st.pid == p][0] for p in src]).value_counts().to_dict())
meta["splits"] = {s: dict(n=len(v), pids=[int(p) for p in v], sha256=hashlib.sha256(",".join(map(str, v)).encode()).hexdigest())
                  for s, v in st.split_pids.items()}
ref = (st.tr.part == "reference").values
full = Ridge(alpha=10.0).fit(st.E[ref], st.rating[ref])
meta["models"] = {"ridge_full": dict(C.model_meta("ridge_full", full, ref.sum()), used_for="paper results (detect_lib.py lines 31-32) and the audit of the old ranking experiment"),
                  "ridge_srcout": dict(st.ridge_meta, used_for="v4 ranking experiment only")}
meta["code_sha256"] = {f: sha(f"{SAD}/rank_v4/code/{f}") for f in ["common.py", "exp.py", "precheck.py", "pilot.py", "diag_prior.py", "freeze_meta.py", "test_common.py", "test_rules.py"]}
meta["protocol_sha256"] = sha(f"{SAD}/rank_v4/PROTOCOL_v4.2.md")
repo = f"{SAD}/../repo_sad"
meta["repo_commit"] = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
meta["env"] = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, machine=platform.machine())
json.dump(meta, open(OUT, "w"), indent=1)
print(json.dumps({k: (v if k not in ("splits",) else {s: (x["n"], x["sha256"][:12]) for s, x in v.items()}) for k, v in meta.items()
                  if k not in ("data_files_sha256", "code_sha256")}, indent=1, default=str)[:4000])
