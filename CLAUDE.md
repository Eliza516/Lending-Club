# CLAUDE.md — Lending Club Credit Scoring

Guidance for Claude Code sessions working in this repository.

## Project

Predict **loan default** on the Lending Club accepted-loans dataset
(Kaggle: `wordsforthewise/lending-club`) from **application-time features only**.
Binary classification: `1 = Charged Off / Default`, `0 = Fully Paid`.

This is a university midterm for *Statistical Learning* (DS66A / DS66B). The course
brief lives in `Midterm Projects môn Học thống kê (Statistical Learning) - DS66A &
DS66B-2.xlsx`.

### Hard constraints from the brief

- **All writing in English**, 100% — markdown cells, figure captions, report, slides.
- **Every figure and table needs a caption AND a discussion** of what it shows.
  Figures with no surrounding discussion lose marks.
- **Every metric used must be explainable** — do not add a metric to the comparison
  table without a markdown cell defining it and saying when it is appropriate.
- **Groups sharing this dataset must share one train/test split.** See below.
- Cite every reference in the body of the report, not just in a bibliography.
- Prefer tables and figures over walls of text.

## Layout

```
notebooks/01_eda.ipynb        Load → clean → OOT split → feature engineering → EDA
notebooks/02_modeling.ipynb   Models → calibration → thresholds → evaluation
data/raw/                     Kaggle CSV or generated synthetic sample (git-ignored)
data/processed/clean.parquet  Hand-off from notebook 01 to 02 (git-ignored)
artifacts/figures/            Saved PNGs (git-ignored)
artifacts/metrics.csv         Model comparison table (git-ignored)
artifacts/split_manifest.json Auditable record of the shared split (git-ignored)
```

## Notebook-first rule

**Do not add a `src/` package or standalone `.py` modules.** The course deliverable is
a notebook. New pipeline code goes into cells in the two existing notebooks. Helper
functions are defined in a cell near their first use, not in an imported file.

## Setup and run

```bash
pip install -r requirements.txt
jupyter lab
# Run notebooks/01_eda.ipynb top to bottom, then notebooks/02_modeling.ipynb.
```

Notebook 01 falls back to a **generated synthetic sample** when the real CSV is absent,
so both notebooks always run end to end. Synthetic results are for smoke-testing the
pipeline only — never quote them as findings.

---

## Leakage rules (the most important section)

Two distinct failure modes. Both are graded, and both are easy to reintroduce.

### 1. Target leakage — features unknowable at application time

Dropped by name in notebook 01 via `LEAKAGE_COLUMNS`. Never reintroduce any of:

| Columns | Why they leak |
|---|---|
| `recoveries`, `collection_recovery_fee` | Only nonzero *after* a charge-off — near-perfect predictors of the label |
| `total_pymnt*`, `total_rec_*` | Cumulative repayment; a defaulted loan repays less |
| `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d` | Payment history, post-origination |
| `out_prncp*` | Outstanding principal — a function of the outcome |
| `last_fico_range_*` | FICO re-pulled *during* the loan; moves with the default itself |
| `debt_settlement_flag*`, `settlement_*`, `hardship_*` | Exist only for distressed loans |
| `funded_amnt*` | Post-decision funding outcome; use `loan_amnt` (the request) instead |
| `loan_status` | The target |

Sanity check: a test ROC-AUC near 0.99 means a leakage column survived. The honest
band on this dataset is roughly **0.68–0.72**.

### 2. Train–test contamination — transforms fitted on data they should not see

**Split first, transform second.** Anything that *learns a parameter from the data* is
fitted on training rows only.

| Operation | Where it may run | Reason |
|---|---|---|
| `%` / `" months"` stripping, `emp_length` map, date parsing | full frame | row-wise, deterministic |
| Ratios (`loan_to_income`, `fico_avg`, `credit_history_years`) | full frame | row-wise, deterministic |
| Target binarization, leakage-column drop, duplicate drop | full frame | fixed rules, no statistics |
| Outlier clipping on `annual_inc` / `dti` | full frame, **fixed domain bounds only** | percentile bounds would learn from test |
| `revol_util` bucketing | full frame, **fixed edges 0/25/50/75/100** | quantile edges would learn from test |
| High-missing column screening | threshold from **train rows only**, applied to both | a missing rate is a statistic |
| Median / mode imputation | inside `Pipeline`, fit on train | learns a statistic |
| `StandardScaler` | inside `Pipeline`, fit on train | learns mean and std |
| `OneHotEncoder` vocabulary, `min_frequency` | inside `Pipeline`, fit on train | learns categories and frequencies |
| Target / WOE encoding | inside `Pipeline`, fit on train | worst offender — encodes the label |

Consequences to preserve:

- All preprocessing lives in a `ColumnTransformer` inside a `Pipeline`. `X_test` is only
  ever passed to `.transform()`, never `.fit()`.
- Cross-validation runs on the `Pipeline` object, so the preprocessor is re-fitted inside
  each fold.
- Calibration (`CalibratedClassifierCV`) is fitted on a slice of the **training** data,
  never on test rows.
- The decision threshold is chosen on training/validation scores and applied unchanged
  to test. Tuning it on test scores is the same family of leakage.
- EDA and feature selection use **pre-cutoff rows only**, so choices made by eye do not
  smuggle test information into the model.

---

## The shared split — out-of-time, not random

The split is **out-of-time (OOT)**: sorted by `issue_d`, loans issued **before
`2016-01-01` are train**, on or after are test.

Why not a random split: default is a forecasting problem. Applicants arrive in time
order, and Lending Club's product mix, credit policy and macroeconomic backdrop drift
across 2007–2018. A random split lets the model train on 2018 loans to predict 2015
ones, which flatters every metric relative to how a scorecard is actually deployed.

A frozen date is also a *better* shared protocol than a seed — it depends on no library
RNG, so other groups reproduce it exactly regardless of implementation.

Rules:

- **Never change the cutoff** without regenerating `artifacts/split_manifest.json` and
  telling the other groups on this dataset.
- Cross-validate with **`TimeSeriesSplit` ordered by `issue_d`**, never shuffled
  `KFold` / `StratifiedKFold` — shuffled folds reintroduce exactly the look-ahead the
  OOT split removes.
- The random stratified split (`random_state=42`, `test_size=0.2`) is computed **only**
  as a reference number, to report the OOT-vs-random gap. It is not the headline result.
- OOT metrics should come out **lower** than random-split metrics. If they come out
  higher, something is wrong — investigate, do not report it.
- Assert `train.issue_d.max() < test.issue_d.min()` — a one-line guard against
  look-ahead.

## Limitations that must stay in the write-up

Do not delete these from the conclusion of notebook 02. They are the difference between
a defensible report and an overclaiming one.

1. **`issue_d` is a proxy for application time.** It is the loan *issue* date. Using it
   for `credit_history_years` and for the OOT cutoff is legitimate and standard practice
   on this dataset, but the dataset never records the application/decision moment. Any
   application-to-issuance lag is invisible, so features are dated slightly later than a
   real scorecard would see them.
2. **Maturity / right-censoring bias.** Keeping only terminal statuses discards `Current`
   loans. 36- and 60-month loans issued near the end of the window have not matured, so
   the post-cutoff test set is enriched with loans that resolved early —
   disproportionately early charge-offs. Its default rate is not the true one.
3. **Accepted-loans-only selection bias.** Training data covers applicants Lending Club
   already approved. This estimates default risk *conditional on acceptance*, not for the
   through-the-door population (the reject-inference problem).
4. **Population drift is measured, not corrected.** A deployed model would need periodic
   refitting.

## Conventions

- Python >= 3.10. `RANDOM_SEED = 42` everywhere a seed is needed.
- Save every figure to `artifacts/figures/` with `plt.savefig(..., dpi=150,
  bbox_inches="tight")`, and give it a caption cell plus a discussion cell.
- Markdown cells in English.
- **Clear all notebook outputs before committing** so diffs stay reviewable:
  `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`
- Never commit `data/` or `artifacts/` — both are git-ignored.
