"""Scenario runner: see detect_lib.py for scenarios and detectors. Usage: python detect.py <work_dir> <out_json>"""
import sys, json, numpy as np, importlib.util
from sklearn.metrics import roc_auc_score, average_precision_score
spec = importlib.util.spec_from_file_location("detect_lib", __file__.replace("detect.py", "detect_lib.py"))
dm = importlib.util.module_from_spec(spec); spec.loader.exec_module(dm)
OUT = sys.argv[2]; build, scores = dm.build, dm.scores
SCEN = [("tr", "flip"), ("tr", "llm"), ("tr", "offtopic"), ("tr", "mixed"), ("en", "cg"), ("en", "flip"), ("en", "offtopic"), ("en", "mixed")]
res = {}
for lang, kind in SCEN:
    for seed in range(5):
        d = build(lang, kind, seed)
        sc = scores(d, lang); y = d["y"]; k = int(y.sum())
        for m, s in sc.items():
            top = np.argsort(-s)[:k]
            r = res.setdefault(f"{lang}-{kind}", {}).setdefault(m, {"auroc": [], "auprc": [], "p_at_k": []})
            r["auroc"].append(roc_auc_score(y, s)); r["auprc"].append(average_precision_score(y, s)); r["p_at_k"].append(y[top].mean())
        print(lang, kind, seed, len(y), k, flush=True)
json.dump(res, open(OUT, "w"), indent=1)
for sc_, ms in res.items():
    print("\n==", sc_)
    for m, v in sorted(ms.items(), key=lambda kv: -np.mean(kv[1]["auprc"])):
        print(f"  {m:30s} AUROC {np.mean(v['auroc']):.3f}  AUPRC {np.mean(v['auprc']):.3f}±{np.std(v['auprc']):.3f}  P@k {np.mean(v['p_at_k']):.3f}")
