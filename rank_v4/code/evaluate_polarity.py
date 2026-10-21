"""Evaluates the blind source-polarity check with the frozen rule (protocol 14.7): per family, share of 'expected'
over all 100 sampled texts ('belirsiz' counts as not expected); pass if >= 0.80.
Usage: python evaluate_polarity.py <filled_csv> <key_csv> <out_json> <filled_xlsx_sha256>"""
import sys, json, hashlib, csv
import common as C
filled, key, out, xsha = sys.argv[1:5]
lab = {r["id"]: r["label"] for r in csv.DictReader(open(filled, encoding="utf-8"))}
K = list(csv.DictReader(open(key, encoding="utf-8")))
assert set(lab) == {r["id"] for r in K} and len(K) == 2 * C.POLARITY_N
res = {}
for fam in ("A1", "A2"):
    rows = [r for r in K if r["family"] == fam]
    mapped = [C.polarity_map(fam, lab[r["id"]]) for r in rows]
    share, ok = C.polarity_result(mapped)
    by_cat = {}
    for r, m in zip(rows, mapped):
        d = by_cat.setdefault(r["category"], {"expected": 0, "not_expected": 0, "unclear": 0}); d[m] += 1
    res[fam] = dict(n=len(rows), expected=mapped.count("expected"), not_expected=mapped.count("not_expected"), unclear=mapped.count("unclear"),
                    share_expected=share, threshold=C.POLARITY_THRESHOLD, passed=bool(ok),
                    status=("confirmatory-eligible (other conditions still apply)" if ok else "exploratory"), by_category=by_cat)
res["note"] = "A3 uses the A1 source pool but is descriptive; it is not classified by this check."
res["hashes"] = dict(filled_xlsx=xsha, filled_csv=hashlib.sha256(open(filled, "rb").read()).hexdigest(),
                     key_csv=hashlib.sha256(open(key, "rb").read()).hexdigest(),
                     common_py=hashlib.sha256(open(C.__file__, "rb").read()).hexdigest())
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
print(json.dumps({f: {k: v for k, v in res[f].items()} for f in ("A1", "A2")}, indent=1, ensure_ascii=False))
