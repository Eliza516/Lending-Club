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

**No data? The notebooks still run.** If the file is missing, notebook 01 generates a
synthetic sample with the same schema, dtypes and plausible value ranges, so the whole
pipeline executes end to end. Synthetic numbers are for smoke-testing only — never quote
them as findings.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Run **`notebooks/01_eda.ipynb` first**, then **`notebooks/02_modeling.ipynb`** — the
first writes `data/processed/clean.parquet`, which the second reads.

## Layout

```
notebooks/01_eda.ipynb          Load → clean → OOT split → features → EDA figures
notebooks/02_modeling.ipynb     Models → calibration → thresholds → evaluation
data/raw/                       Kaggle CSV or generated synthetic sample   (git-ignored)
data/processed/clean.parquet    Hand-off between the two notebooks         (git-ignored)
artifacts/figures/              Saved PNGs                                 (git-ignored)
artifacts/metrics.csv           Model comparison table                     (git-ignored)
artifacts/split_manifest.json   Auditable record of the shared split       (git-ignored)
CLAUDE.md                       Conventions, leakage rules, split protocol
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
4. Drift is measured by the OOT design but not corrected for.

See `CLAUDE.md` for the full leakage rules and project conventions.
