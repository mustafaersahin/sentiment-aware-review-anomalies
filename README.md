# When Stars and Words Disagree: sentiment-aware anomaly detection in product reviews

Code and results accompanying the manuscript *"When Stars and Words Disagree: Sentiment-Aware Anomaly Detection in
Product Reviews and Its Effect on Product Rankings"* (B. Erşahin and M. Erşahin).

## Data
- Turkish: Trendyol product reviews (Hugging Face `Doukan/trendyol-turkish-product-reviews`, CC BY-NC 4.0).
- English: Fake Reviews Dataset of Salminen et al. (2022) (Hugging Face `theArijitDas/Fake-Reviews-Dataset`).
Review texts are not redistributed; `prepare.py` rebuilds the corpora deterministically (seed 7).

## Pipeline
    python code/prepare.py <data_dir> work            # corpora and reference/evaluation split
    python code/gen_fake_tr.py work                   # LLM-written Turkish promotional reviews (gpt-4.1-nano; output in data/)
    python code/encode.py work emb                    # multilingual sentence embeddings
    python code/encode.py work sent                   # pretrained sentiment scores
    python code/detect.py work work/detect.json       # 8 scenarios x 5 seeds, all detectors
    python code/ranking.py work work/ranking.json     # product-ranking manipulation experiment
    python code/natural_audit.py work code            # audit sample of natural inconsistencies (gpt-4.1-mini labels)
    python code/make_outputs.py work <paper_dir>      # tables and figures

`detect_lib.py` contains the scenario construction and all detectors, including SAAD (maximum of robust z-scores of the
sentiment-rating gap, product-context distance and style outlierness).

## Files
- `data/llm_promotional_reviews_tr.csv` – the 1,427 LLM-written Turkish reviews used as manipulations (product id, text).
- `data/natural_audit_llm_labels.csv` – review ids, sample (top-200 by sentiment gap / random-200) and LLM labels.
- `results/` – detection and ranking results (all seeds).

## Licence
Code: MIT.
