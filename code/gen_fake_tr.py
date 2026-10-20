"""Generate LLM-written promotional Turkish reviews for the evaluation products (gpt-4.1-nano).
Each call returns 6 short 5-star reviews for one product, imitating real marketplace reviews.
The API key is injected by the environment's network proxy. Usage: python gen_fake_tr.py <work_dir>"""
import sys, re, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
W = sys.argv[1]
c = pd.read_parquet(f"{W}/tr_corpus.parquet")
ev = c[c.part == "evaluation"].drop_duplicates("pid")[["pid", "product_name", "category"]]
PROMPT = ("Bir e-ticaret sitesinde satılan şu ürün için müşteri ağzından 6 farklı, 5 yıldızlı, olumlu Türkçe yorum yaz. "
          "Gerçek kullanıcı yorumları gibi kısa ve günlük dilde olsun (3 ile 25 kelime arası, uzunlukları farklı), "
          "bazılarında kargo veya paketlemeden de söz edilebilir. Numara, tırnak veya açıklama ekleme; her yorumu ayrı satıra yaz.\n"
          "Ürün: {name}\nKategori: {cat}")
def call(row):
    body = {"model": "gpt-4.1-nano", "temperature": 1.0, "max_tokens": 400,
            "messages": [{"role": "user", "content": PROMPT.format(name=row.product_name, cat=row.category)}]}
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for a in range(4):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=60)); break
        except Exception:
            time.sleep(2 ** a)
    lines = [re.sub(r"^\d+[.)]\s*", "", l.strip(" -•\"'")).strip() for l in r["choices"][0]["message"]["content"].split("\n") if len(l.strip()) > 2]
    u = r["usage"]; cost = (u["prompt_tokens"] * 0.10 + u["completion_tokens"] * 0.40) / 1e6
    return [{"pid": int(row.pid), "text": l, "cost": cost / max(1, len(lines))} for l in lines[:6]]
with ThreadPoolExecutor(8) as ex:
    out = [x for lst in ex.map(call, list(ev.itertuples())) for x in lst]
pd.DataFrame(out).to_parquet(f"{W}/tr_llm_fake.parquet")
print(len(out), "reviews; cost USD", round(sum(o["cost"] for o in out), 4))
