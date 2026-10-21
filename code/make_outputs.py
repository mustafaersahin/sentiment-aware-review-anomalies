"""Tables, figure and text macros for the manuscript. Usage: python make_outputs.py <work_dir> <paper_dir>"""
import sys, os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
W, P = sys.argv[1], sys.argv[2]
D = json.load(open(f"{W}/detect.json"))
# legacy v1 ranking experiment (legacy/ranking_v1): optional, not used by the revised manuscript
R = json.load(open(f"{W}/ranking.json")) if os.path.exists(f"{W}/ranking.json") else None
SC = ["tr-flip", "tr-llm", "tr-offtopic", "tr-mixed", "en-flip", "en-cg", "en-offtopic", "en-mixed"]
HEAD = ["Flip", "LLM", "Off-topic", "Mixed", "Flip", "CG", "Off-topic", "Mixed"]
M = [("IF (TF-IDF)", "Isolation Forest, TF-IDF"), ("IF (text)", "Isolation Forest, embedding"), ("LOF (text)", "LOF, embedding"),
     ("OCSVM (text)", "One-class SVM, embedding"), ("AE (text)", "Autoencoder, embedding"), ("IF (text+rating)", "Isolation Forest, embedding + rating"),
     ("Sentiment gap (pretrained)", "Sentiment gap, pretrained model"), ("Sentiment gap (rating model)", "Sentiment gap, rating model"),
     ("Sentiment gap (ensemble)", "Sentiment gap, ensemble"), ("Context distance", "Product/category context distance"), ("Style (IF)", "Style outlierness"),
     ("SAAD w/o style", "SAAD-core (sentiment + context)"), ("SAAD (proposed)", "SAAD-full (+ style)")]
mean = lambda m, s, k="auprc": np.mean(D[s][m][k]); sd = lambda m, s, k="auprc": np.std(D[s][m][k])
mac = []
def table(metric, label, cap):
    L = [r"\begin{table*}[t]", r"\centering", r"\caption{" + cap + "}", r"\label{" + label + "}", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{@{}l" + "r" * 8 + "@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{Turkish (Trendyol)} & \multicolumn{4}{c}{English (Fake Reviews Dataset)} \\", r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
         "Detector & " + " & ".join(HEAD) + r" \\", r"\midrule"]
    best = {s: max(mean(m, s, metric) for m, _ in M) for s in SC}
    for k, (m, nm) in enumerate(M):
        if k in (6, 11): L.append(r"\midrule")
        cells = []
        for s in SC:
            v = mean(m, s, metric); c = f"{v:.3f}"
            cells.append(r"\textbf{" + c + "}" if abs(v - best[s]) < 1e-9 else c)
        L.append(nm + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(f"{P}/tables/{label.split(':')[1]}.tex", "w").write("\n".join(L) + "\n")
table("auprc", "tab:auprc", "Area under the precision--recall curve (AUPRC; mean over five injection seeds; standard deviations $\\leq 0.02$) for each anomaly scenario at about 5\\% contamination, where a random ranking scores about 0.05. Bold: best detector per scenario. The first block uses review content only; the second, sentiment--rating consistency and context; the third, the proposed combinations.")
table("auroc", "tab:auroc", "AUROC (mean over five seeds) for each anomaly scenario.")
# fusion ablation
AB = [("SAAD (proposed)", "Max of robust $z$-scores (SAAD-full)"), ("SAAD w/o style", "\\quad without style (SAAD-core)"), ("SAAD w/o context", "\\quad without context"),
      ("SAAD w/o sentiment", "\\quad without sentiment"), ("SAAD, sum fusion", "Sum of robust $z$-scores"), ("SAAD, rank fusion", "Max of rank-normalised scores")]
L = [r"\begin{table}[t]", r"\centering", r"\caption{Ablation of the combination: AUPRC in the mixed scenarios and the Turkish LLM scenario (mean over five seeds).}", r"\label{tab:ablation}", r"\small",
     r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", r"Combination & TR mixed & EN mixed & TR LLM \\", r"\midrule"]
for m, nm in AB: L.append(f"{nm} & {mean(m,'tr-mixed'):.3f} & {mean(m,'en-mixed'):.3f} & {mean(m,'tr-llm'):.3f} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(f"{P}/tables/ablation.tex", "w").write("\n".join(L) + "\n")
# ranking table (legacy v1 only)
RM = [("No filtering", "No filtering"), ("IF (text)", "Isolation Forest, embedding"), ("OCSVM (text)", "One-class SVM, embedding"), ("AE (text)", "Autoencoder, embedding"),
      ("Sentiment gap (pretrained)", "Sentiment gap, pretrained"), ("Sentiment gap (rating model)", "Sentiment gap, rating model"), ("Sentiment gap (ensemble)", "Sentiment gap, ensemble"),
      ("Context distance", "Context distance"), ("SAAD w/o style", "SAAD-core"), ("SAAD (proposed)", "SAAD-full")]
L = [r"\begin{table*}[t]", r"\centering", r"\caption{Product-ranking manipulation on the Turkish evaluation products (192 products with at least 40 reviews; 20 promoted and 20 demoted per seed; mean over five seeds). Each detector removes the 5\% highest-scoring reviews before re-ranking. $\tau$: Kendall correlation with the clean ranking; rank error: mean absolute rank displacement of attacked products; NDCG@20 with respect to the clean ranking; removed: share of injected and of genuine reviews removed. Without an attack, $\tau = 1$ and the rank error is 0.}",
     r"\label{tab:ranking}", r"\small", r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule", r"Filter & $\tau$ & Rank error & NDCG@20 & Injected removed & Genuine removed \\", r"\midrule"]
for m, nm in (RM if R is not None else []):
    v = R["5pct"][m]; L.append(f"{nm} & {np.mean(v['tau']):.3f} & {np.mean(v['rank_err']):.1f} & {np.mean(v['ndcg20']):.3f} & {100*np.mean(v['injected_removed']):.1f}\\% & {100*np.mean(v['genuine_removed']):.1f}\\% \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
if R is not None:
    open(f"{P}/tables/ranking.tex", "w").write("\n".join(L) + "\n")
# natural audit: macros are written by human_audit.py (tables/audit_macros.tex)
# figure: AUPRC of groups
fig, ax = plt.subplots(figsize=(9, 3.2))
grp = [("IF (text)", "Isolation Forest (content)", "#999999"), ("Sentiment gap (ensemble)", "Sentiment gap", "#1f77b4"), ("Context distance", "Context distance", "#2ca02c"),
       ("Style (IF)", "Style", "#ff7f0e"), ("SAAD w/o style", "SAAD-core", "#d62728"), ("SAAD (proposed)", "SAAD-full", "#9467bd")]
x = np.arange(len(SC)); w = 0.13
for k, (m, lab, col) in enumerate(grp):
    ax.bar(x + (k - 2.5) * w, [mean(m, s) for s in SC], w, label=lab, color=col)
ax.axhline(0.05, color="k", ls="--", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([("TR " if s.startswith("tr") else "EN ") + h for s, h in zip(SC, HEAD)], fontsize=8)
ax.set_ylabel("AUPRC"); ax.legend(fontsize=7, ncol=3, loc="upper right"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(f"{P}/figs/auprc.pdf"); plt.savefig(f"{P}/figs/auprc.png", dpi=200)
# macros
r5 = R["5pct"] if R is not None else None
for nm, m in ([("NoF", "No filtering"), ("RM", "Sentiment gap (rating model)"), ("Core", "SAAD w/o style"), ("IFt", "IF (text)"), ("Ens", "Sentiment gap (ensemble)"), ("Full", "SAAD (proposed)")] if R is not None else []):
    mac.append(f"\\newcommand{{\\rk{nm}Err}}{{{np.mean(r5[m]['rank_err']):.1f}}}"); mac.append(f"\\newcommand{{\\rk{nm}Tau}}{{{np.mean(r5[m]['tau']):.3f}}}")
    mac.append(f"\\newcommand{{\\rk{nm}Gen}}{{{100*np.mean(r5[m]['genuine_removed']):.1f}}}"); mac.append(f"\\newcommand{{\\rk{nm}Inj}}{{{100*np.mean(r5[m]['injected_removed']):.1f}}}")
for nm, m, s in [("ensTrFlip", "Sentiment gap (ensemble)", "tr-flip"), ("ensEnFlip", "Sentiment gap (ensemble)", "en-flip"), ("coreTrMix", "SAAD w/o style", "tr-mixed"),
                 ("coreEnMix", "SAAD w/o style", "en-mixed"), ("fullTrMix", "SAAD (proposed)", "tr-mixed"), ("fullEnMix", "SAAD (proposed)", "en-mixed"),
                 ("ensTrMix", "Sentiment gap (ensemble)", "tr-mixed"), ("ensEnMix", "Sentiment gap (ensemble)", "en-mixed"), ("styleTrLlm", "Style (IF)", "tr-llm"),
                 ("ctxEnOff", "Context distance", "en-offtopic"), ("ctxTrOff", "Context distance", "tr-offtopic"), ("tfidfEnCg", "IF (TF-IDF)", "en-cg"),
                 ("ifTrMix", "IF (text)", "tr-mixed"), ("ifEnMix", "IF (text)", "en-mixed")]:
    mac.append(f"\\newcommand{{\\{nm}}}{{{mean(m, s):.3f}}}")
open(f"{P}/tables/macros.tex", "w").write("\n".join(mac) + "\n"); print("\n".join(mac))
