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
  The one exception is `docs/methodology_walkthrough.vi.md`, an internal-reading
  translation. Its English counterpart is the deliverable; edit both together or the
  translation goes stale.
- **Every figure and table needs a caption AND a discussion** of what it shows.
  Figures with no surrounding discussion lose marks.
- **Every metric used must be explainable** — do not add a metric to the comparison
  table without a markdown cell defining it and saying when it is appropriate.
- **Groups sharing this dataset must share one train/test split.** See below.
- Cite every reference in the body of the report, not just in a bibliography.
- Prefer tables and figures over walls of text.

## Layout

The project follows an **11-step predictive-modeling process**, four stages, one notebook
per stage. Run them in order.

```
notebooks/01_problem_framing.ipynb          Stage A - Steps 1-2   (decisions, no data)
notebooks/02_data_and_features.ipynb        Stage B - Steps 3-7   (data -> clean.parquet)
notebooks/03_modeling_and_evaluation.ipynb  Stage C - Steps 8-10  (models, eval, explain)
notebooks/04_packaging_and_monitoring.ipynb Stage D - Step 11     (package, monitor)

data/raw/                     Kaggle CSV or generated synthetic sample (git-ignored)
data/processed/clean.parquet  Hand-off from notebook 02 to 03        (git-ignored)
artifacts/figures/            Saved PNGs, numbered 01-19             (git-ignored)
artifacts/metrics.csv         Model comparison incl. both baselines  (git-ignored)
artifacts/split_manifest.json Auditable record of the shared split   (git-ignored)
artifacts/model.joblib        Complete fitted pipeline               (git-ignored)
artifacts/model_card.json     Provenance, metrics, versions          (git-ignored)

docs/related_work.md          Benchmark comparison vs published projects (committed)
docs/methodology_walkthrough.md  Decision-by-decision account of how the model was
                              built, and why                             (committed)
docs/methodology_walkthrough.vi.md  Vietnamese translation of the above, for internal
                              reading only -- the English file is the deliverable
```

### Where does new code go?

| Step | Topic | Lives in |
|---|---|---|
| 1 | Problem summary + input/output spec, framing, cost of errors, incumbent baseline | NB01 §1 |
| 2 | Problem type, target rule, metric choice, baseline spec | NB01 §2 |
| 3 | Data dictionary, granularity, load, target, leakage block | NB02 §3 |
| 4 | The out-of-time split + manifest | NB02 §4 |
| 5 | EDA (training rows only) | NB02 §5 |
| 6 | Logic checks, cleaning, missingness, column screening | NB02 §6 |
| 7 | Feature engineering | NB02 §7 |
| 8 | Baselines, models, CV, overfitting check, tuning | NB03 §8 |
| 9 | Calibration, threshold, metrics, error analysis, fairness | NB03 §9 |
| 10 | Permutation importance, SHAP, partial dependence | NB03 §10 |
| 11 | Packaging, round-trip, PSI, retraining policy | NB04 §11 |

**Execution order vs step order.** NB02 runs Step 6 and 7 *before* Step 5: EDA cannot
precede cleaning because `int_rate` is the string `"13.56%"` in the raw file, and several
figures use engineered features. What matters is that EDA precedes every modelling
decision and sees training rows only. The process is a loop, not a line.

## Notebook-first rule

**Do not add a `src/` package or standalone `.py` modules.** The course deliverable is
a notebook. New pipeline code goes into cells in the four existing notebooks — see the
table above for which one. Helper functions are defined in a cell near their first use,
not in an imported file.

## Setup and run

```bash
pip install -r requirements.txt
jupyter lab
# Run notebooks 01 -> 02 -> 03 -> 04 in order, each top to bottom.
```

Notebook 02 falls back to a **generated synthetic sample** when the real CSV is absent,
so every notebook always runs end to end. Synthetic results are for smoke-testing the
pipeline only — never quote them as findings.

---

## Leakage rules (the most important section)

Two distinct failure modes. Both are graded, and both are easy to reintroduce.

### 1. Target leakage — features unknowable at application time

Dropped by name in notebook 02 via `LEAKAGE_COLUMNS`. Never reintroduce any of:

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

## Process discipline (the 11-step method)

The project follows an 11-step predictive-modeling process. Three ordering rules matter
more than the rest, because breaking them invalidates results silently:

1. **Baselines before models, models before tuning** (NB03 §8). Two baselines exist: B0
   majority class, and **B1 `sub_grade` — the incumbent underwriting rule**. The project's
   claim is the *lift over B1*, not the absolute AUC. If a model cannot beat B1, report
   that; do not tune until it appears to.
2. **The split is decided before exploring** (NB02 §4, before §5). An analyst who has
   studied the test period has already leaked it through their own choices.
3. **The test set is opened once** (NB03 §9.4). Thresholds are chosen on the calibration
   slice; tuning is scored with `TimeSeriesSplit` on training data only.

Also required and easy to drop when editing:

- Every figure needs a caption, a discussion, **and an `Action →` line**. A figure that
  leads to no action gets cut, not kept.
- The fairness screen (NB03 §9.7) is a **proxy** screen — the data has no protected
  attributes. Never describe it as a compliance audit.
- **The 151 → 30 column reduction is rule-based (NB02 §3.2), not a hand-written list.**
  `classify_column()` puts every column in the file into exactly one bucket, and an
  assertion fails the notebook if any column is left `UNCLASSIFIED`. `APPLICATION_COLUMNS`
  and `LEAKAGE_COLUMNS` are **derived** from it — never edit them directly. A new column
  from a data refresh must be given a bucket, and its timing answered in the §3.3 data
  dictionary, before it can be used.
- The largest bucket is `sparse_pre2012_bureau` (50 columns) — real application-time
  bureau fields that are empty before ~2012, which is inside our training window. They are
  excluded by scope, not because they leak. Restricting training to 2012+ vintages would
  make most of them usable, and pairs with the matured-vintage fix in
  `docs/related_work.md`.
- Target encoding is deliberately unused. If added, it must be `TargetEncoder` inside the
  `Pipeline` so it fits out-of-fold.

## Competitive positioning

`docs/related_work.md` benchmarks this project against six published Lending Club projects
and one peer-reviewed paper, with citations. Keep it current when results change. Key
facts from it that constrain what we may claim:

- The published AUC range on this dataset is **0.678–0.735**, which confirms the
  0.68–0.72 band above. Out-of-time projects score *lower* than random-split ones, as our
  §9.8 predicts.
- Models beat Lending Club's own grade by only **+0.012 to +0.018 AUC** in the two
  published projects that measured it. A large lift over B1 is a red flag, not a win.
- **Fairness screening is our clearest differentiator** — none of the surveyed projects
  does it.
- We are **behind** published work on: right-censoring (others restrict to matured
  vintages), cost realism (ours is a placeholder), confidence intervals on the lift, and
  enforcing leakage rules in code rather than in documentation. Do not overclaim.

## Limitations that must stay in the write-up

Do not delete these from the conclusion of notebook 03. They are the difference between
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
