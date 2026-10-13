# CLAUDE.md — Lending Club Credit Scoring

Guidance for Claude Code sessions working in this repository.

## Project

Predict **loan default** on the Lending Club accepted-loans dataset
(Kaggle: `wordsforthewise/lending-club`) from **application-time features only**.
Binary classification: `1 = Charged Off`, `0 = Fully Paid`.

University midterm for *Statistical Learning* (DS66A / DS66B). The course brief is the
`Midterm Projects ... .xlsx` file in the repo root.

## Guiding principle: simple, and every claim comes from code

The author is a student who must be able to explain every line.

- **Every number or insight in a notebook is printed by a code cell.** Never type a
  statistic, a default rate, an AUC or a ranking into markdown by hand, and never quote
  numbers from outside the project (papers, blogs, other repos). Markdown explains *what
  we do and why*; the discussion after each figure/table is a code cell that prints
  f-string sentences from the computed objects.
- If an expectation is not borne out by the data (e.g. out-of-time AUC not lower than
  random-split AUC), the notebook must say so; do not rewrite the narrative to fit.
- Prefer plain pandas / scikit-learn calls over clever code. No SHAP, no custom
  classes, no hidden helper modules. If a technique needs more than a short paragraph
  to explain to a classmate, leave it out.
- The notebooks are committed **with outputs**: the outputs are the evidence. Re-running
  them top to bottom is part of any change.

### Hard constraints from the brief

- **All writing in English**, 100% — markdown cells, figure captions, README.
- **Every figure and table needs a caption AND a discussion** (caption in markdown,
  discussion as a printing code cell that ends with an `Action →` line when a decision
  follows).
- **Every metric used must be defined** (Table 1 of notebook 03) with when it is
  appropriate and its weakness.
- **Groups sharing this dataset share one train/test split** — see below.
- Prefer tables and figures over walls of text.

## Layout

```
notebooks/01_problem_framing.ipynb          Steps 1-2   decisions, no data
notebooks/02_data_and_features.ipynb        Steps 3-7   data -> clean.parquet
notebooks/03_modeling_and_evaluation.ipynb  Steps 8-10  baselines, models, evaluation, explanation
notebooks/04_packaging_and_monitoring.ipynb Step 11     reload, drift, retraining rule, model card

data/raw/accepted_2007_to_2018Q4.csv.gz  Kaggle file (git-ignored, required)
data/processed/clean.parquet, features.json   hand-off 02 -> 03/04 (git-ignored)
artifacts/                               figures, metrics.csv, model.joblib, threshold.json,
                                         split_manifest.json, model_card.json (git-ignored)
```

**Notebook-first rule:** no `src/` package, no standalone `.py` modules. Helper functions
are defined in a cell near their first use.

There is no synthetic-data fallback: the real file is required so that every printed number
is a real finding.

## Setup and run

```bash
pip install -r requirements.txt
jupyter lab            # run 01 -> 02 -> 03 -> 04, each top to bottom
```

Notebook 03 has a `STEP` constant (1 = all rows, 10 = every 10th row) for quick trial runs.
Commit only runs with `STEP = 1`.

## Leakage rules

### 1. Target leakage — only columns known at application time
Notebook 02 §3.2 lists the kept columns with *why each is known at application time*.
Everything else in the 151-column file is dropped. §4.3 runs a code smell test (single-column
ROC-AUC on training rows) that shows what post-origination columns (`total_pymnt`,
`recoveries`, `last_pymnt_amnt`, `last_fico_range_high`) look like: far above any
application-time column. A new column must be added to the §3.2 dictionary with its timing
reason before it can be used. Never use `loan_status` or any payment/recovery/settlement/
hardship column as a feature.

### 2. Train–test contamination — split first, transform second
Anything that *learns a number from data* (median, mean, std, category list, any target
encoding) lives inside the `Pipeline` in notebook 03 and is fitted on training rows only.
Notebook 02 only applies fixed row-by-row rules (string parsing, ratios, logs, fixed bounds).

- `X_test` is only passed to `.predict*` / `.transform`, never `.fit`.
- Cross-validation is run on the `Pipeline` object, so preprocessing is re-fitted per fold.
- Use `clone(model)` inside `make_pipeline` so two pipelines never share one fitted model.
- The decision threshold is chosen on a validation slice (the latest 20% of training rows
  by date), never on test scores.
- EDA and every data decision use training rows only. The only place all years are shown is
  the label-maturity diagnostic (NB02 §4.2), which informs a limitation, not a modelling choice.

## The shared split — out-of-time

Loans issued **before `2016-01-01` are train**, on or after are test. A frozen date depends
on no random seed, so other groups reproduce it exactly.

- **Never change the cutoff** without regenerating `artifacts/split_manifest.json` and telling
  the other groups. The manifest holds counts and an MD5 of the sorted training ids.
- Cross-validate with `TimeSeriesSplit`, never shuffled K-fold.
- The random stratified split (`random_state=42`, `test_size=0.2`) exists only as a reference
  (NB03 §9.7). The notebook prints whether out-of-time AUC really is lower; if it is not,
  investigate before reporting.
- `assert train.issue_d.max() < test.issue_d.min()` stays in the notebooks.

## Process discipline

1. **Baselines before models, models before tuning.** B0 majority class; **B1 = Lending Club
   `sub_grade`** — the incumbent. The claim is the *lift over B1* (with a bootstrap range),
   not the absolute AUC. NB03 §8.8 shows the model with and without `grade`/`int_rate`.
2. **The split is made before exploring** (NB02 §4 before §5).
3. **The test set is opened once** (NB03 §9.2). Model choice comes from training-only
   cross-validation; later test-set tables are analysis, not selection.

## Limitations that must stay in the conclusion of notebook 03

1. `issue_d` is a proxy for application time.
2. Right-censoring: unfinished loans are dropped, so late test years are measured on an
   early-resolving subset (evidence: NB02 Table 4).
3. Accepted loans only (no reject inference).
4. Drift is measured (NB03 §9.6/§9.8, NB04), not corrected.
5. Threshold uses F1 because the data contains no real costs.
6. No fairness analysis — the data has no protected attributes. Do not describe anything
   as a fairness or compliance audit.

## Conventions

- Python >= 3.10, `RANDOM_SEED = 42` wherever a seed is needed.
- Save every figure with `plt.savefig(..., dpi=150, bbox_inches="tight")` into `artifacts/figures/`.
- Never commit `data/` or `artifacts/` (git-ignored).
- `docs/*.md` belong to the previous version of the project (they quote external numbers and
  describe SHAP/PSI steps that no longer exist). Do not cite them; they are pending removal.
