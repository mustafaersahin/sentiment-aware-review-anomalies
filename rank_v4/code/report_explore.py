"""Renders the EXPLORATORY summary as markdown tables (no ranking of methods, no winner).
Usage: python report_explore.py <summary_json> <out_md>"""
import sys, json
s = json.load(open(sys.argv[1]))
M = ["B0", "B2p", "R2", "R3", "R4", "gap_rm", "gap_ens", "saad_core", "gap_cal", "LAF"]
r3 = lambda x: "–" if x is None else f"{x:.3f}"
ci = lambda d, f="{:.2f}": "–" if d is None or d.get("lo") is None else f"{f.format(d['mean'])} [{f.format(d['lo'])}, {f.format(d['hi'])}]"
L = [f"# Sıralama moderasyonu – KEŞİFSEL test analizi", "",
     f"**{s['label']}**", "", f"Plan kaydı sha256: `{s['plan_record_sha256']}`; model: {s['model']['label']} ({s['model']['train_rows']} satır, {s['model']['sha256'][:12]}…).", "",
     "Aralıklar: R = 10 tekrar üzerinden t aralığı (D_rel: delete-one-replicate jackknife SE ile t aralığı). Yalnızca Monte Carlo belirsizliği; doğrulayıcı anlamlılık, üstünlük veya genelleme değildir. Yöntemler sabit bir sırada listelenmiştir, sıralama değildir.", ""]
for fam, als in [("A1", [0.1, 0.2]), ("A2", [0.1, 0.2, 0.3]), ("A3", [0.1, 0.2])]:
    L.append(f"## {fam}")
    for a in als:
        for rho in [0.5, 1.0, 2.0, 4.0]:
            b = s["cells"][f"{fam}|{a}|{rho}|B0"]
            tag = "stres testi (ρ = 4)" if rho == 4.0 else ("ana eğri; §4 stres hücresi (b/(n+k) > 0,25)" if b["budget_share_max"] > 0.25 else "ana eğri")
            L += ["", f"### {fam}, α = {a}, ρ = {rho:g} — {tag}", f"Hedef {b['n_targets']} (L ≥ 1: {b['n_targets_L_ge1']}); en büyük bütçe payı {b['budget_share_max']:.2f}; kırpılan koşu {b['clipped_runs']}.", "",
                  "| Yöntem | D | D_rel | TRR | TRR (ağırlıksız) | Yakalama | 1★ sil. | 2★ sil. | 3★ sil. | 5★ sil. | AUC (1★ havuzu) | Zenginleşme | Sınır (exact/near) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for m in M:
                k = f"{fam}|{a}|{rho}|{m}"
                if k not in s["cells"]:
                    continue
                c = s["cells"][k]; dr = s["D_rel"][k]
                d_rel = f"{dr['estimate']:.3f} [{dr['lo']:.3f}, {dr['hi']:.3f}]"
                bd = c["boundary"]
                L.append(f"| {m} | {ci(c['D'], '{:.1f}')} | {d_rel} | {ci(c['TRR'], '{:.3f}')} | {ci(c['TRR_unweighted'], '{:.3f}')} | {ci(c['capture'], '{:.3f}')} | {r3(c['del_1'])} | {r3(c['del_2'])} | {r3(c['del_3'])} | {r3(c['del_5'])} | {ci(c.get('auc_1star'), '{:.3f}')} | {ci(c.get('enrichment'), '{:.2f}')} | {bd.get('exact', 0)}/{bd.get('near', 0)} |")
L += ["", "## Mekanizma karşıtlığı: yakalama(A2) − yakalama(A3)", "", "| α | ρ | " + " | ".join(M) + " |", "|---|---|" + "---|" * len(M)]
for a in [0.1, 0.2]:
    for rho in [0.5, 1.0, 2.0, 4.0]:
        mc = lambda m: s["mechanism_contrast"].get(f"{a}|{rho}|{m}") or s["mechanism_contrast"].get(f"{a}|{rho:g}|{m}")   # keys hold rho as written in the plan
        L.append(f"| {a} | {rho:g} | " + " | ".join(ci(mc(m), "{:.3f}") for m in M) + " |")
L += ["", "## FF (yanlış işaretleme) — kontrol ürünleri, eşleşen hedef sayısıyla ağırlıklı", "", "Skor tabanlı yöntemler FF'de deterministiktir (MC belirsizliği yok); rastgele yöntemler R = 10 üzerinden.", ""]
for fam in ["A1", "A2"]:
    for a in ([0.1, 0.2] if fam == "A1" else [0.1, 0.2, 0.3]):
        L += ["", f"### FF {fam} (A3 aynı kontrolleri paylaşır), α = {a}", "", "| ρ | Yöntem | D | Puan sapması | TRR_c | 1★ sil. | Kontrol / çift |", "|---|---|---|---|---|---|---|"]
        for rho in [0.5, 1.0, 2.0, 4.0]:
            for m in M:
                c = s["ff"].get(f"{fam}|{a}|{rho}|{m}")
                if c:
                    L.append(f"| {rho:g} | {m} | {c['D_weighted']['mean']:.2f} | {c['score_dev_weighted']:.4f} | {r3(c['TRR'])} | {r3(c['del_1'])} | {c['controls']} / {c['pairs']} |")
L += ["", "## Ağırlık yoğunlaşması (yalnızca tanı; etkin örneklem büyüklüğü değildir)", ""] + \
     [f"- {k}: (ΣL)²/ΣL² = {v['value']:.2f}; L ≥ 1 hedef {v['targets_L_ge1']}; en büyük L {v['L_max']}" for k, v in s["weight_concentration"].items()]
open(sys.argv[2], "w").write("\n".join(L) + "\n")
