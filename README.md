# Sentiment–Rating Gaps for Detecting Manipulated Product Reviews

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23531378.svg)](https://doi.org/10.5281/zenodo.23531378)

Code and results accompanying the manuscript *"Sentiment–Rating Gaps for Detecting Manipulated Product Reviews:
Evidence from Turkish and English Data"* (B. Erşahin and M. Erşahin).

> **Status:** revised version for the manuscript, archived as release `v2.0.0` (commit `e5f9210`):
> [10.5281/zenodo.23531379](https://doi.org/10.5281/zenodo.23531379); all versions: [10.5281/zenodo.23531378](https://doi.org/10.5281/zenodo.23531378).
> Release `v2.0.1` adds `requirements.txt`, environment notes and the `OPENAI_API_KEY` option; code paths and results are unchanged.
> The state before the revision is tagged `v1.0-initial` (commit `f30c2b1`). See *Versions* below.

## Data
- Turkish: Trendyol product reviews, Hugging Face `Doukan/trendyol-turkish-product-reviews`, revision
  `3b390598dac913449a75b624a06e985a1c5d726c`, file `trendyol_reviews_20251212_000835.csv` (CC BY-NC 4.0).
  The released file contains 582,601 reviews of 476 distinct product pages (89–99 per category); the dataset card states
  500 products (100 per category) and does not document the difference. All 476 products enter the prepared corpus.
- English: Fake Reviews Dataset of Salminen et al. (2022) as distributed in Hugging Face `theArijitDas/Fake-Reviews-Dataset`,
  revision `cffa887a877db747e540f15eb891f0d18070b994` (40,526 reviews; 40,427 after keeping the ten categories with more
  than 1,000 reviews and removing duplicate texts).

Review texts, embeddings and per-review scores are not redistributed; `code/prepare.py` rebuilds the corpora
deterministically (seed 7) and the other scripts regenerate everything else.

## Environment
Python 3.13 on Linux x86-64, CPU only (four vCPUs, 15 GB memory); package versions in `requirements.txt`
(`pip install -r requirements.txt`). `gen_fake_tr.py` and `natural_audit.py` call the OpenAI API and read the key from
`OPENAI_API_KEY`; their outputs are distributed in `data/`, so the API is not needed to reproduce the results.

## Detection experiments and natural-inconsistency audit (main results of the manuscript)
    python code/prepare.py <data_dir> work            # corpora and reference/evaluation split
    python code/gen_fake_tr.py work                   # LLM-written Turkish promotional reviews (gpt-4.1-nano; output in data/)
    python code/encode.py work emb                    # multilingual sentence embeddings
    python code/encode.py work sent                   # pretrained sentiment scores
    python code/detect.py work work/detect.json       # 8 scenarios x 5 seeds, all detectors
    python code/natural_audit.py work code            # audit sample of natural inconsistencies (gpt-4.1-mini labels)
    cp data/natural_audit_human_labels.csv work/human_audit.csv
    python code/human_audit.py work paper/tables/audit_macros.tex   # human vs LLM agreement
    python code/make_outputs.py work <paper_dir>      # tables and figures

To use the distributed LLM labels instead of calling the API, copy `data/natural_audit_llm_labels.csv` to
`work/natural_audit.csv`; `human_audit.py` accepts its `llm_label` column. `make_outputs.py` builds the legacy ranking
table only if `work/ranking.json` (legacy v1) exists.

## Exploratory ranking analysis — `rank_v4/` (EXPLORATORY)
A pre-registered confirmatory comparison was planned but not carried out: the pre-specified attack-size criterion
failed on the validation products (`rank_v4/out/selection_record_A1.json`). The whole ranking analysis is therefore
**exploratory**: fixed test cohort of 128 products, synthetic single-product attacks, flagged product assumed known,
intervals describe Monte Carlo variation of the attack draws only. See `rank_v4/README.md`.

## Versions
- `legacy/ranking_v1/` – the earlier ranking experiment (`ranking.py`, `ranking.json`) of commit `f30c2b1`. It is
  superseded and not reported in the revised manuscript; kept for transparency (see its README).
- `rank_v4/` – protocol, code, frozen records and outputs of the exploratory ranking analysis.

## Files
- `data/llm_promotional_reviews_tr.csv` – the 1,427 LLM-written Turkish reviews used as manipulations (product id, text).
- `data/natural_audit_llm_labels.csv` – review ids, sample (top-200 by sentiment gap / random-200) and LLM labels.
- `data/natural_audit_human_labels.csv` – blind human labels (consistent / inconsistent / unclear) for the same 400 reviews.
- `results/detection.json` – detection results (all scenarios, detectors and seeds; aggregate metrics).

## Citation
See `CITATION.cff`. Cite all versions: https://doi.org/10.5281/zenodo.23531378 (release v2.0.0: https://doi.org/10.5281/zenodo.23531379).

## Licence
Code: MIT.
