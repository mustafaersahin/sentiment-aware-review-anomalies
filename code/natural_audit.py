"""Audit of natural (non-injected) rating-text inconsistencies in the Turkish evaluation reviews.
Sample: 200 reviews ranked highest by the sentiment-gap ensemble and 200 random reviews (seed 11).
Each is labelled by gpt-4.1-mini (temperature 0) as consistent / inconsistent / unclear with its star rating.
Usage: python natural_audit.py <work_dir> <code_dir>"""
import os, sys, json, time, urllib.request, importlib.util, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
HEADERS = {"Content-Type": "application/json"}
if os.environ.get("OPENAI_API_KEY"):          # otherwise the key must be injected by a network proxy
    HEADERS["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
W, C = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("dl", C + "/detect_lib.py"); dm = importlib.util.module_from_spec(spec); spec.loader.exec_module(dm)
g = dm.tr[dm.tr.part == "evaluation"]; gi = g.index.values
r = g.rating.values.astype(float)
gap = (dm.rz(np.abs((r - 1) / 4 - dm.S["tr"][gi])) + dm.rz(np.abs(r - dm.rid_tr.predict(dm.E["tr"][gi])))) / 2
rng = np.random.default_rng(11)
top = np.argsort(-gap)[:200]; rand = rng.choice(np.setdiff1d(np.arange(len(g)), top), 200, replace=False)
PROMPT = ("Aşağıdaki ürün yorumunun metni, kullanıcının verdiği yıldız puanıyla tutarlı mı? Metindeki genel memnuniyete bak; "
          "kargo veya satıcı şikâyetleri de memnuniyetsizlik sayılır.\nPuan: {r}/5\nYorum: {t}\n"
          'Yalnızca JSON döndür: {{"label": "consistent" | "inconsistent" | "unclear"}}')
def ask(i):
    body = {"model": "gpt-4.1-mini", "temperature": 0, "seed": 7, "max_tokens": 20, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": PROMPT.format(r=int(r[i]), t=g.text.values[i][:1500])}]}
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(), headers=HEADERS)
    for a in range(4):
        try: x = json.load(urllib.request.urlopen(req, timeout=60)); break
        except Exception: time.sleep(2 ** a)
    u = x["usage"]
    return json.loads(x["choices"][0]["message"]["content"]).get("label", "unclear"), (u["prompt_tokens"] * 0.4 + u["completion_tokens"] * 1.6) / 1e6
idx = np.concatenate([top, rand])
with ThreadPoolExecutor(8) as ex: out = list(ex.map(ask, idx))
lab = [o[0] for o in out]
res = pd.DataFrame({"rid": g.rid.values[idx], "sample": ["top200"] * 200 + ["random200"] * 200, "rating": r[idx], "gap": gap[idx], "label": lab})
res.to_csv(f"{W}/natural_audit.csv", index=False)
print(res.groupby("sample").label.value_counts().unstack().fillna(0)); print("cost USD", round(sum(o[1] for o in out), 4))
