# Exploratory ranking-moderation analysis (v4) — EXPLORATORY, NOT CONFIRMATORY

Protocol: `PROTOCOL_v4.2.md` (sections 14–16 hold the frozen decisions and the exploratory plan).

## What happened
1. Pre-checks, pilot and unit tests (`code/precheck.py`, `code/pilot.py`, `code/test_*.py`; outputs in `out/`).
2. Blind human polarity check of the attack source texts (`code/make_polarity_sheet.py`, `code/evaluate_polarity.py`):
   A1 0.94, A2 0.93 expected polarity (threshold 0.80) — `out/polarity_result.json`.
3. Validation selection for the planned confirmatory family A1 (`code/select_validation.py`): the alpha rule failed
   (median normalised displacement 0.032 and 0.048 < 0.08), so no confirmatory test was run —
   `out/selection_record_A1.json` (+ `.sha256`; the hash file lists the original local path) and `out/selection_record_A1_addendum.md`.
4. Exploratory plan frozen before any test-split computation — `out/exploratory_plan_record.json` (+ `.sha256`).
5. Exploratory test analysis (`code/explore_test.py`, `code/report_explore.py`) — `out/explore_test_summary.json`,
   `out/explore_test_report.md`, row-level metrics `out/explore_test_rows*.csv.gz`, hashes `out/explore_test_outputs.sha256`.

## How to read the results
Fixed cohort of 128 Turkish test products; synthetic attacks on one flagged product at a time; the flagged product is
assumed to be known; ten replicates. Intervals describe only the variation of the synthetic attack draws, not sampling
uncertainty or significance. No method is declared better than another. `gap_cal` is a variant added during protocol
revision and is not part of the manuscript's claims.

## Polarity check: what is shared
- `in/polarity_labels.csv`: opaque id and the annotator's label only (no texts).
- Not shared: the annotator workbook and the id–text sheet (they contain review texts) and the id–family key.
  The key is not needed: `code/make_polarity_sheet.py` regenerates the sample and the key deterministically from the
  corpus (SeedSequence 20261021), after which `code/evaluate_polarity.py` reproduces `out/polarity_result.json`
  (check the regenerated files against the hashes in `out/polarity_sample_report.json`).

## Layout expected by the scripts
The scripts take a base directory with `work/` (from `code/prepare.py`, `code/encode.py`), `code/detect_lib.py` and
`rank_v4/`; the repository root has this layout once `work/` exists. `precheck.py` and `freeze_meta.py` also hash
files of the authors' working copy (`paper/`, raw data) and are kept as records of that run.

Models: `ridge_srcout` (16,222 reviews of the 167 reference products not used as text sources) is used here only;
the manuscript's detection results use `ridge_full` (23,234 reviews of all 238 reference products). Hashes in
`out/frozen_metadata.json`.
