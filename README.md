# Lending Club Credit Scoring

Predicting loan default from **application-time features only**, on the Lending Club
accepted-loans dataset (2007–2018).

Midterm project for *Statistical Learning* (DS66A / DS66B).

## Problem

Binary classification. Given the information available when a loan application is
scored — requested amount, term, interest rate, grade, employment, income, DTI, FICO
band, credit-file summary — predict whether the loan will eventually be **charged off**.

| | |
|---|---|
| **Target** | `1` = `Charged Off` / `Default`, `0` = `Fully Paid` |
| **Excluded** | `Current`, `Late`, `In Grace Period`, `Issued` — outcome not yet known |
| **Class balance** | ≈ 20 % default |
| **Split** | Out-of-time at `issue_d = 2016-01-01` (see below) |
| **Headline metrics** | ROC-AUC, PR-AUC, KS, Gini, Brier |

## Data

Source: [Kaggle — wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

The raw file (`accepted_2007_to_2018Q4.csv.gz`, ~1.6 GB uncompressed) is **not** in this
repository. Download it into `data/raw/`:

```bash
mkdir -p data/raw

# Option A — Kaggle CLI (needs ~/.kaggle/kaggle.json)
kaggle datasets download -d wordsforthewise/lending-club -p data/raw --unzip

# Option B — download from the Kaggle web UI and move the file yourself:
#   data/raw/accepted_2007_to_2018Q4.csv.gz
```

**No data? The notebooks still run.** If the file is missing, notebook 02 generates a
synthetic sample with the same schema, dtypes and plausible value ranges, so the whole
pipeline executes end to end. Synthetic numbers are for smoke-testing only — never quote
them as findings.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

`xgboost` and `shap` are optional — both imports are guarded, so the notebooks run end to
end without them.

## Method: the 11-step predictive-modeling process

Four notebooks, four stages. **Run them in order.**

| Notebook | Stage | Steps | Produces |
|---|---|---|---|
| `01_problem_framing.ipynb` | A — before code | 1. Framing · 2. Type, target, metric | The decisions everything else depends on |
| `02_data_and_features.ipynb` | B — data | 3. Understand · 4. Split · 5. EDA · 6. Clean · 7. Features | `clean.parquet`, `split_manifest.json`, figures 01–07 |
| `03_modeling_and_evaluation.ipynb` | C — model | 8. Build · 9. Evaluate · 10. Explain | `metrics.csv`, figures 08–17 |
| `04_packaging_and_monitoring.ipynb` | D — production | 11. Package, deploy, monitor | `model.joblib`, `model_card.json`, figures 18–19 |

Notebook 01 loads no data at all — the process puts problem framing *before touching
code*, and that is where the incumbent baseline and the cost of errors get decided.

### Baselines

The project's claim is not its absolute AUC but its **lift over the incumbent**:

| # | Baseline | What it is |
|---|---|---|
| B0 | Majority class | Predict "never defaults" — the 80 %-accurate useless model |
| **B1** | **`sub_grade` as a risk score** | **Lending Club's existing underwriting** — the real bar |

### What each stage adds beyond a typical notebook

- **Data dictionary with a timing test** — every column answers *"is this known at
  application time?"* before it can be used as a feature.
- **Data-logic checks** — inverted FICO bands, credit lines opened after the loan,
  non-positive income.
- **Missing-reason taxonomy** — blanks meaning "never happened" get an indicator column
  instead of being median-imputed away.
- **Overfitting check** — train-vs-CV gap per model, which is how the ensembles get caught.
- **Error analysis by segment** — including by issue quarter, which sets the retraining
  cadence.
- **Fairness screen** — decline rate, FPR and FNR across proxy groups with a
  disparate-impact ratio. Proxies only; the data has no protected attributes.
- **SHAP and partial dependence** — per-applicant attribution and the shape of each effect.
- **PSI drift monitoring and a retraining policy** — with explicit numeric triggers.

## Layout

```
notebooks/01_problem_framing.ipynb          Stage A - decisions, loads no data
notebooks/02_data_and_features.ipynb        Stage B - data, split, EDA, cleaning, features
notebooks/03_modeling_and_evaluation.ipynb  Stage C - models, evaluation, explanation
notebooks/04_packaging_and_monitoring.ipynb Stage D - packaging, drift, retraining policy
data/raw/                       Kaggle CSV or generated synthetic sample   (git-ignored)
data/processed/clean.parquet    Hand-off from notebook 02 to 03            (git-ignored)
artifacts/figures/              Saved PNGs, numbered 01-19                 (git-ignored)
artifacts/metrics.csv           Model comparison incl. both baselines      (git-ignored)
artifacts/split_manifest.json   Auditable record of the shared split       (git-ignored)
artifacts/model.joblib          Complete fitted pipeline                   (git-ignored)
artifacts/model_card.json       Provenance, metrics, versions, limitations (git-ignored)
CLAUDE.md                       Conventions, leakage rules, split protocol
docs/related_work.md            Benchmark comparison vs published projects
```

## Train / test split protocol

Groups working on this dataset must use the **same** split. Ours is **out-of-time**:

> Sort by `issue_d`. Loans issued **before 2016-01-01** are training data; loans issued
> **on or after** are test data.

Rationale: default prediction is a forecasting problem. Applicants arrive in time order,
and the platform's product mix, credit policy and macroeconomic backdrop drift across
2007–2018. A random split lets a model train on 2018 loans to predict 2015 ones,
inflating every metric relative to deployment. A frozen date is also more reproducible
than a random seed, since it depends on no library RNG.

The train/test **proportion is determined by origination volume, not fixed at 80/20**.
Lending Club's volume grew by orders of magnitude across the window, so a 2016 cutoff
splits the resolved loans roughly in half. That is a property of the data, not a knob to
tune — moving the cutoff to hit a target ratio would mean choosing the split by looking at
the outcome. `artifacts/split_manifest.json` records the actual counts for each run.

Cross-validation on the training side uses `TimeSeriesSplit`, never shuffled folds.
A random stratified split (`random_state=42`, `test_size=0.2`) is computed **only** as a
reference, so the report can quantify how optimistic random splitting is.

`artifacts/split_manifest.json` records the cutoff, per-side row counts, class balance,
date ranges and a hash of the test index, so the split is auditable and shareable.

## Metrics, and when to use them

| Metric | What it measures | When to prefer it |
|---|---|---|
| **ROC-AUC** | Ranking quality across all thresholds | Threshold-free comparison; but optimistic under class imbalance because true negatives dominate |
| **PR-AUC** (average precision) | Precision–recall trade-off for the positive class | Better than ROC-AUC here — defaults are the rare, expensive class |
| **KS statistic** | Max separation between the good and bad score CDFs | Industry standard in credit scoring; how well the scorecard separates populations |
| **Gini** (`2·AUC − 1`) | Rescaled ROC-AUC | Conventional in credit risk reporting; same information as AUC |
| **Brier score** | Mean squared error of predicted probabilities | Probabilities are the product — a model that ranks well but is miscalibrated prices risk wrongly |
| **Precision / Recall / F1** | Performance at one chosen threshold | Only meaningful once a threshold is justified; 0.5 is arbitrary at 20 % prevalence |

Thresholds are chosen on training/validation scores (F1-optimal, Youden-J, and a
cost-based rule with explicit false-negative and false-positive costs), then applied
unchanged to test.

## Known limitations

1. `issue_d` is a proxy for the application date — the true decision moment is not
   recorded, so features are dated slightly later than a live scorecard would see them.
2. Right-censoring: filtering to terminal statuses drops unmatured loans, enriching the
   late test period with loans that resolved early.
3. Accepted loans only — no rejected applicants, so this is risk *conditional on
   acceptance* (reject inference).
4. Drift is measured by the OOT design and monitored via PSI in notebook 04, but the
   model is not drift-corrected — it needs scheduled retraining.
5. The fairness screen uses proxies (region, income band, housing); the data contains no
   protected attributes, so it is indicative, not a compliance audit.

See `CLAUDE.md` for the full leakage rules and project conventions, and
`docs/related_work.md` for how this project compares against published work on the same
dataset — including where it is ahead, and where it is behind.
