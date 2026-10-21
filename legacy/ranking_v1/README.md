# LEGACY – earlier ranking-manipulation experiment (v1, superseded)

`ranking.py` and `ranking.json` are the ranking experiment of commit `f30c2b1` (moved here from `code/` and `results/`
without modification). They are **not reported** in the revised manuscript.

Why superseded (audit in `../../rank_v4/out/diag_prior.json`, which reproduces this experiment exactly):
- the filtering budget (5% of all reviews) was about 2.6 times the number of injected reviews;
- the demotion attack consisted of one-star reviews whose texts were consistent with the rating, so filters that remove
  one-star reviews repaired the ranking; a budget-matched rule removing one-star reviews gave a similar result;
- the injected reviews were copies of existing reviews.

The replacement analysis (`../../rank_v4/`) is exploratory. To rerun this script, copy it back to `code/` (it imports
`code/detect_lib.py`).
