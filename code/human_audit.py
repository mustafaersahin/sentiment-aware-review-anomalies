"""Agreement of the human labels with the LLM labels in the natural-inconsistency audit.
Usage: python human_audit.py <work_dir> <out_macros>"""
import sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

W, OUT = sys.argv[1], sys.argv[2]
c = pd.read_parquet(f"{W}/tr_corpus.parquet")
m = pd.read_csv(f"{W}/natural_audit.csv").merge(pd.read_csv(f"{W}/human_audit.csv"), on="rid").merge(c[["rid", "text"]], on="rid")
m["trunc"] = m.text.astype(str).str.contains("Devamını Oku")
pct = lambda x: f"{100 * x.mean():.1f}"
mac = {}
for s, g in m.groupby("sample"):
    s = s.replace("200", "")
    mac[f"audit{s}Inc"] = pct(g.human == "inconsistent")
    mac[f"audit{s}Unc"] = pct(g.human == "unclear")
    mac[f"audit{s}Trunc"] = pct(g.trunc)
    mac[f"audit{s}IncFull"] = pct(g[~g.trunc].human == "inconsistent")
top = m[m["sample"] == "top200"].sort_values("gap", ascending=False)
mac["audittopFiftyInc"] = pct(top.head(50).human == "inconsistent")
b = m[(m.human != "unclear") & (m.label != "unclear")]
mac["auditKappa"] = f"{cohen_kappa_score(b.human, b.label):.2f}"
mac["auditAgree"] = pct(b.human == b.label)
mac["auditNbin"] = str(len(b))
mac["auditAUROC"] = f"{roc_auc_score(m.human == 'inconsistent', m.gap):.2f}"
mac["auditEnrich"] = f"{float(mac['audittopInc']) / float(mac['auditrandomInc']):.1f}"
c = pd.read_parquet(f"{W}/tr_corpus.parquet")
mac["corpusTrunc"] = pct(c.text.astype(str).str.contains("Devamını Oku"))
open(OUT, "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
print(mac)
