# Lending Club Credit Scoring

Predict whether a Lending Club loan will be **charged off**, using only information known
when the loan is decided. Midterm project for *Statistical Learning* (DS66A / DS66B).

Every number in the notebooks is printed by code from the real Kaggle file. Nothing is
copied from outside sources.

## Data

Kaggle: [wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
Put `accepted_2007_to_2018Q4.csv.gz` in `data/raw/` (it is git-ignored, ~390 MB compressed).
There is no synthetic fallback: the notebooks stop with a clear message if the file is missing.

## Run

```bash
pip install -r requirements.txt
jupyter lab        # run notebooks 01 -> 02 -> 03 -> 04, each top to bottom
```

Notebook 03 takes about 35 minutes on 4 CPU cores. Set `STEP = 10` at the top of it for a
10x faster trial run (do not quote results from a trial run).

## What each notebook does

| Notebook | Steps | What you get |
|---|---|---|
| `01_problem_framing` | 1-2 | Problem, input/output, baselines, metric choice, split plan. No data loaded. |
| `02_data_and_features` | 3-7 | Data dictionary, target, **out-of-time split**, leakage smell test, EDA on training rows, cleaning, features |
| `03_modeling_and_evaluation` | 8-10 | Baselines, 3 models, time-aware cross-validation, tuning, threshold, test evaluation, bootstrap, explanation |
| `04_packaging_and_monitoring` | 11 | Save/reload check, scoring one application, drift checks, retraining rule, model card |

## Design choices

- **Target:** `1` = Charged Off, `0` = Fully Paid. Unfinished loans are dropped.
- **Split:** loans issued before 2016-01-01 are train, on or after are test. Fixed and shared by
  all groups; `artifacts/split_manifest.json` holds counts and an MD5 of the training ids.
- **Leakage:** only columns known at application time are kept (table in notebook 02 §3.2).
  A code test (notebook 02 §4.3) shows post-loan columns such as `last_fico_range_high` predict
  far better than any legitimate column - the signature of leakage.
- **No contamination:** imputing, scaling and encoding live inside a scikit-learn `Pipeline`
  fitted on training rows only; cross-validation is `TimeSeriesSplit`.
- **Baselines:** B0 = predict nobody defaults; **B1 = Lending Club's own sub-grade**. The claim
  is the lift over B1, not the raw AUC.
- **Models:** logistic regression, random forest, histogram gradient boosting. The final one is
  chosen by training-only cross-validation, not by looking at the test set.

## Results (from notebooks 03 and 04, full data)

| | ROC-AUC (test) | PR-AUC (test) |
|---|---|---|
| B0 majority class | 0.500 | 0.224 |
| B1 Lending Club sub-grade | 0.687 | 0.365 |
| Logistic regression | 0.704 | 0.387 |
| Random forest | 0.704 | 0.396 |
| **Histogram gradient boosting (chosen)** | **0.711** | **0.404** |

- Lift over B1: **+0.024 AUC**, 95% bootstrap range [+0.023, +0.025] (notebook 03 §9.3).
- Without Lending Club's own `grade` and `int_rate`, the cross-validated lift over B1 falls from
  +0.023 to +0.013 (§8.8): part of the result depends on Lending Club's own assessment.
- Out-of-time AUC (0.711) is lower than a random split's (0.720), as expected (§9.7).
- The model predicts fewer defaults than happened on average (0.182 vs 0.224): use the score to
  **rank** applicants, not as an exact probability (§9.6).
- Choosing among the three models matters little (0.007 AUC spread on test); the gap to B1 is larger.

## Limitations

1. `issue_d` (loan issue date) stands in for the application date.
2. Unfinished loans are dropped, so late test years are measured on an early-resolving subset
   (notebook 02, Table 4).
3. Only accepted loans are in the data (no reject inference).
4. Drift is measured, not corrected (notebook 04). A few extreme `loan_to_income` values in the
   test period slip through the cleaning rule; notebook 04 §11.3 documents this.
5. The decision threshold maximises F1 because the data has no real cost of errors.
6. No fairness analysis: the data has no protected attributes.

## Repository notes

- `CLAUDE.md` holds the working rules for Claude Code sessions in this repo.
- `docs/` contains write-ups from an earlier version of the project. They mention steps the
  notebooks no longer contain (SHAP, PSI) and numbers that do not come from the current
  notebooks, so treat them as outdated until rewritten.
