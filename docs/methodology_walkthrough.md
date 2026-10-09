# How This Model Was Built — A Credit Risk Analyst's Walkthrough

A first-person account of the reasoning behind every decision in this project: what I
looked at, what I concluded from it, what I did, and what each choice cost.

This is the *why* document. The notebooks are the *what*. Read this first if you need to
defend, extend, or audit the model.

---

## Phase 0 — Before opening the file

The temptation with a famous dataset is to load it and start plotting. I did not, because
the questions that decide whether a credit model is usable cannot be answered from the
data.

**What I needed settled first:**

| Question | My answer | Why it had to come first |
|---|---|---|
| Who acts on the score? | Credit decisioning: approve/decline, price tier, exposure | If no decision changes, stop — there is no project |
| Predict what, when? | `P(charged off)` at the moment of application | This fixes which columns are legal. Everything downstream depends on it |
| What does being wrong cost? | FN (missed default) ≫ FP (declined good applicant) | Decides the metric and the cut-off, not the algorithm |
| What already exists? | **Lending Club's own `grade` / `sub_grade`** | This is the incumbent. It is the thing I must beat |

The fourth question is the one most projects skip, and skipping it is how people end up
proud of an AUC of 0.73 without ever checking that the platform's existing 7-level grade
already achieves ~0.68 on its own. **A model is worth its complexity only against the
rule it replaces.**

**Decision:** two baselines fixed before any modelling — B0 majority class (sanity floor),
B1 `sub_grade` as a raw score (the real bar).

I also wrote the cost ratio down as `FN:FP = 4:1` and flagged it as a placeholder. Naming
a guess as a guess is what stops it quietly becoming a finding.

---

## Phase 1 — First contact: reading the file as a risk analyst

The raw file is **2,260,701 rows × 151 columns**, issued 2007-06 to 2018-12 (33 rows have
an unparseable `issue_d` and are dropped). Every number in this walkthrough comes from a
full run of notebooks 02–04 on that file. Three things I check before anything else, in
this order.

**1. What is a row?** One *funded* loan application. Not one borrower (repeat borrowers
exist, no stable key to group them). Not one application (rejected applicants are in a
different file, with no outcome). This one sentence already implies the
accepted-loans-only selection bias that limits every claim I can make later.

**2. What is the time axis?** `issue_d`, monthly granularity, 2007–2018. Volume grows by
orders of magnitude across the window. That immediately told me a random split would be
wrong — more on this in Phase 3.

**3. What does the outcome column actually contain?** `loan_status` has seven-ish values,
not two. `Current`, `Late (31–120 days)`, `In Grace Period`, `Issued` are **unresolved** —
the outcome is not yet known. Only `Fully Paid`, `Charged Off` and `Default` are terminal.

That third observation is where the first real decision lives.

---

## Phase 2 — Column triage: the timing test

151 columns, and most of them are poison. I applied one question to every single column:

> *On the day the application arrives, does this field already have a value?*

The answer sorts the file into two piles.

**Legal (~30 columns):** the loan request (amount, term, purpose), the applicant
(income, employment, housing, state), and the credit bureau pull (FICO band, DTI,
utilisation, open accounts, delinquencies, credit history length).

**Poison (~40 columns):** anything that exists only because money was already lent.

| Column family | Why it is fatal |
|---|---|
| `recoveries`, `collection_recovery_fee` | Nonzero only *after* a charge-off. This is the label wearing a hat |
| `total_pymnt*`, `total_rec_*` | A defaulted loan repays less. This is the label as a continuous variable |
| `last_pymnt_*`, `next_pymnt_d`, `out_prncp*` | Payment behaviour, i.e. the thing being predicted |
| `last_fico_range_*` | FICO **re-pulled during the loan**. It moves *with* the default |
| `hardship_*`, `settlement_*`, `debt_settlement_flag*` | Only populated for distressed loans |
| `funded_amnt*` | Post-decision. Use `loan_amnt` — what the applicant *asked* for |

This is the failure mode that produces the 0.99-AUC notebooks on this dataset. The model
is not predicting default; it is reading a disguised copy of the answer.

**Decision:** I restricted `read_csv(usecols=...)` to the legal list — so the poison never
enters memory — and *also* kept an explicit `LEAKAGE_COLUMNS` drop afterwards as a guard
against a future edit widening the list. Two layers, because this error is silent and
expensive.

I also excluded `emp_title` and `title`: free text, tens of thousands of distinct values,
no ordinal structure, and a natural vehicle for accidental target encoding.

**Sanity rule I wrote into the repo:** the honest band on this dataset is ROC-AUC
≈ 0.68–0.72. Anything near 0.99 means a poison column survived. Published work spans
0.678–0.735 (`docs/related_work.md`), which confirms the band.

---

## Phase 3 — Defining the label, and paying for it

Keep only terminal statuses: `Fully Paid` → 0, `Charged Off`/`Default` → 1. Drop the rest.

This is necessary — you cannot train on an outcome that has not happened. But it is **not
free**, and I made myself write the cost down:

A 36-month loan issued in 2017 cannot have reached maturity by the 2018 Q4 snapshot. It
appears in the data only if it resolved **early**, and the share that has resolved
collapses with vintage age:

| Issue year | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 |
|---|---|---|---|---|---|---|
| Resolved (terminal status) | 100% | 94.7% | 89.2% | 67.5% | 38.2% | **11.4%** |

I expected early resolution to mean early *charge-off*, and therefore an inflated
late-period default rate. The real file says the bias **changes sign with vintage age**:

- **Mid-aged vintages (2016–2017)** are enriched with defaults — test quarters 2016Q2–Q3
  default at 25–26% against 18.4% in training.
- **The newest vintages are enriched with early *prepayments*.** A loan can be paid off in
  its first months, but it cannot be charged off until it has been delinquent for roughly
  120+ days. So 2018Q3 shows a 9.9% default rate and 2018Q4 shows **2.4%**.

**Consequence:** the observed default rate in the test period is **not** the true default
rate in either direction, and the last two quarters are unusable as evidence. Some of the
movement is real drift; some is this artefact. I cannot cleanly separate them.

**Decision:** keep the filter, document the bias as a named limitation, and never quote
the test-period default rate as a cohort rate. (Phase 16 revisits this — it is the one
place where a published project does better than me.)

On the real file this leaves **1,345,350 resolved loans at a 19.96% default rate**
(915,318 unresolved rows dropped), consistent with the surveyed projects. The 2,749
`Does not meet the credit policy` rows are also dropped, which is why the 2007–2010
vintages show resolution rates below 100%.

---

## Phase 4 — Freezing the split *before* looking at anything

This is the step whose ordering matters most and whose violation is least detectable.

**Why out-of-time, not random.** Default prediction is a forecast. The model will score
*next quarter's* applicants. Across 2007–2018 the platform's volume, product mix, credit
policy and macro backdrop all moved. A random split lets the model train on 2018 loans in
order to predict 2015 ones — information no deployed scorecard could ever have. It does
not make the model better; it makes the *measurement* flattering.

**Decision:** sort by `issue_d`, train on everything before `2016-01-01`, test on
everything from that date on. Frozen.

Three design notes:

- **A date beats a seed as a shared protocol.** The course requires groups on this dataset
  to use one split. A cutoff date depends on no library RNG, so it reproduces exactly
  across implementations and versions. `train_test_split(random_state=42)` does not.
- **The resulting ratio is 61/39 (826,604 train / 518,385 test), not 80/20.** Nobody chose
  it — it falls out of the date. In the *raw* file the post-cutoff side is the larger one
  (1.37M of 2.26M loans, 61%), because origination volume grew every year. But only 37.8%
  of post-cutoff loans had resolved by the snapshot, against 93.1% of pre-cutoff loans, so
  the terminal-status filter shrinks the test side far more and flips the ratio. The ratio
  itself is a symptom of the right-censoring in Phase 3. I did *not* move the cutoff to hit
  a nicer ratio — choosing a split by looking at what it produces is choosing by outcome,
  and the cutoff is shared with the other groups on this dataset.
- **I compute a random split too**, but only as a reference number, to quantify how much
  optimism it would have bought. It is never the headline.

**Why before EDA.** An analyst who has studied the test period has already leaked it — not
through code, but through their own choices of feature and model. No metric detects this.
So the split is decided first, and all exploration afterwards sees training rows only.

I wrote `split_manifest.json` recording the cutoff, per-side counts, date ranges, class
balance and a hash of the test index, so the split is auditable and another group can
prove they reproduced it.

> **A bug I caught here.** I originally wrote the manifest right after splitting — but
> later steps still drop rows (duplicates, impossible values). On the real file the
> manifest's counts would then disagree with the exported data, and the downstream
> assertion would fail. It passed on the synthetic sample only because nothing got
> dropped. Fixed: the manifest is written after all row filtering, computed on the final
> row set. The real run confirms the fix mattered: 361 rows are dropped *after* the split
> (Phase 5), and notebook 03's manifest check still passes.

---

## Phase 5 — Cleaning: what the data says that cannot be true

Three distinct jobs here, and conflating them is how mistakes hide.

**(a) Parsing.** The raw file stores `int_rate` as `"13.56%"`, `term` as `" 36 months"`,
`emp_length` as `"10+ years"`, dates as `"Dec-2015"`. Pure string surgery — row-wise,
deterministic, no statistic learned. Safe on the full frame.

**(b) Logic checks.** Before imputing anything, I check the data describes a possible
world:

| Check | What a violation means |
|---|---|
| `earliest_cr_line` after `issue_d` | Credit file opened after the loan. Impossible |
| `fico_range_high` < `fico_range_low` | Inverted band. Impossible |
| `annual_inc` ≤ 0 | Impossible for an approved applicant |
| `open_acc` > `total_acc` | Implausible, not impossible — bureau reporting lag |

I report counts *before* acting, because the count is the diagnosis: a handful of rows is
a data defect; thousands means **my parsing is wrong**, not that Lending Club published
thousands of impossible loans. Hard impossibilities get dropped — there is no honest way
to invent a credit-history start date, and imputing over it would hide a quality signal.

On the real file the counts are reassuringly small: **361 rows with `annual_inc` ≤ 0**
(0.027%, dropped), 2 negative `dti`, 1 row with `open_acc` > `total_acc`, and zero
inverted FICO bands or credit files opened after the loan. Parsing is sound.

**(c) Outliers — and the trap.** The obvious move is to clip at the 1st/99th percentile.
**That is leakage.** A percentile is a statistic computed from the data; using one before
the split lets the test period's income distribution shape how training rows are
transformed. I clip at **fixed domain bounds** instead (`annual_inc` ≤ $1.5M, `dti` ≤ 60)
— limits that come from underwriting plausibility, not from where a quantile happens to
fall. They bite rarely: 123 incomes, 1,716 DTI values and 20 utilisation values are
clipped out of 1.34M rows.

The same logic forced `revol_util` buckets to fixed edges (0/25/50/75/100) rather than
quantile bins.

**(d) Label normalisation.** `home_ownership` carries `ANY`, `NONE` and `OTHER` — three
labels for one residual category, an artefact of the application form changing over the
years (286 `ANY`, 144 `OTHER`, 48 `NONE` on the real file). Left alone they become three
sparse dummies meaning the same thing, and worse,
their relative frequencies drift, so the encoder learns a different vocabulary for the
training period than the test period contains. Collapsed to one.

---

## Phase 6 — Missingness: is the blank telling me something?

The standard move is `fillna(median)` and move on. That destroys information here.

I split missing columns into two kinds:

**Structural — the value does not exist.** `mths_since_last_delinq` is blank for roughly
half of all applicants, and the blank means **"this person has never been delinquent"** —
which is the single most *protective* fact in the file. Median-imputing it says instead
"their last delinquency was a typical number of months ago", which is exactly backwards.

**Not recorded — the value exists but was not captured.** `mort_acc`,
`pub_rec_bankruptcies` are missing for older vintages because the bureau field was not
returned then. The blank says something about the *reporting era*, not the applicant.

**Decision:** for structurally-missing columns, add an explicit `_was_missing` indicator
**before** imputation. The imputer then fills the value; the indicator preserves the fact
of absence, and the model can learn what absence means.

I verified it was worth doing rather than assuming. On the real training rows
`mths_since_last_delinq` is blank for **51.3%** of applicants, and they default at
**17.8%** versus **19.1%** for those with a value. The blank is protective, as the domain
reasoning predicted — though the effect is modest. The `emp_length` blank points the
other way and harder: **23.7%** default when missing versus 18.1% when present, and
`emp_length_was_missing` ends up among the model's top-15 permutation importances
(Phase 14). If the rates had matched I would have dropped the indicator — a column that
adds no information is just variance.

**Column screening.** I drop columns missing in >60% of rows — but the threshold is
measured on **training rows only**, then applied to both sides. A missing rate is a
statistic, and measuring it on the full frame would let the test period vote on which
columns the model may use.

---

## Phase 7 — Features: encoding what an underwriter actually reasons about

A raw column pair does not give a linear model a ratio for free. The features I built are
the quantities a credit officer would compute by hand:

| Feature | Reasoning |
|---|---|
| `loan_to_income` | Leverage — the most interpretable affordability measure there is |
| `installment_to_income` | Payment burden for *this* loan, which `dti` excludes |
| `credit_history_years` | Thin files are riskier. (Uses `issue_d` as a proxy for application date — see Phase 16) |
| `fico_avg` | The bureau reports a band; the midpoint is the usable scalar |
| `is_36_month` | 60-month loans are structurally riskier |
| `grade_ordinal`, `sub_grade_ordinal` | **Ordinal, not one-hot.** The grades are ordered and the default rate is monotone in them — one-hot would throw that away |

All row-wise, so all safe on the full frame.

**What I deliberately did not build: target encoding.** Replacing `addr_state` with the
mean default rate of that state is powerful and is the most dangerous transform available
here. Computed over the rows the model trains on, every row has partly seen its own
answer — a state with one loan gets an encoding equal to that loan's outcome. If it is
ever added, the only safe form is `TargetEncoder` *inside* the `Pipeline`, which fits
out-of-fold. I noted this in the notebook rather than leaving it as a trap.

---

## Phase 8 — Making leakage structurally impossible, not merely avoided

Imputation learns a median. Scaling learns a mean and standard deviation. One-hot encoding
learns a category vocabulary. Every one of them fitted on the full frame is contamination,
and **no metric will reveal it** — the scores just come out quietly too good.

The anti-pattern:

```python
preprocessor.fit(X)                       # <- sees test rows
X_train, X_test = train_test_split(X)     # <- too late
```

**Decision:** every learned transform lives in a `ColumnTransformer` inside a `Pipeline`.
`X_test` then reaches it only through `.transform()`, never `.fit()`. This makes the
correct behaviour *structural* rather than a matter of my discipline on a Friday
afternoon. It also means cross-validation re-fits preprocessing inside each fold instead
of once over all training data.

**How I verified it rather than trusting it:** I compared the imputer's learned medians
against (a) the medians of the fit slice and (b) the medians of the full frame. They match
(a) and differ from (b). If preprocessing had been contaminated, they would have matched
(b). (That audit was run during development, not as a notebook cell.) On the real file the training-only and full-frame medians differ by **$1,000 on
`annual_inc`** and $452 on `revol_bal` — small, which is exactly the point: contamination
of this size moves no headline metric visibly, so only a structural guarantee catches it.

---

## Phase 9 — Baselines first

Before any model:

| Baseline | Test ROC-AUC | Reading |
|---|---|---|
| B0 majority class | 0.5000, PR-AUC 0.2242 (= test prevalence) | The floor. Beating it proves nothing |
| **B1 `sub_grade` (incumbent)** | **0.6871**, PR-AUC 0.3648, KS 0.271 | The real bar |

B1 at 0.687 sits right where the published incumbent benchmarks do (0.679–0.680,
`docs/related_work.md`). Lending Club's own grade is already a good model.

B1 needs no fitting at all — `sub_grade` is already an ordered risk ranking, and every
rank-based metric (AUC, PR-AUC, KS) can consume it directly.

Having this number *before* modelling changes how every later result reads. The question
stops being "is my AUC good?" and becomes "**is my AUC better than the rule the platform
already runs?**"

---

## Phase 10 — Models, simple to complex, and reading the overfitting

Four models, each a full `Pipeline`: logistic regression (the auditable scorecard
baseline), random forest, HistGradientBoosting, XGBoost.

Cross-validation used `TimeSeriesSplit` on chronologically ordered training rows — never
shuffled folds, which would reintroduce precisely the look-ahead the OOT split removes.

I asked for train-fold scores alongside validation scores, because that gap is the
overfitting check:

| Model | Train AUC | CV AUC (± std) | Gap |
|---|---|---|---|
| Logistic Regression | 0.7012 | 0.7194 ± 0.017 | −0.018 |
| Random Forest | 0.7352 | 0.7216 ± 0.016 | +0.014 |
| HistGradientBoosting | 0.7271 | 0.7247 ± 0.018 | +0.002 |
| XGBoost | 0.7409 | 0.7252 ± 0.019 | +0.016 |

**Nobody is memorising.** With ~830k training rows, every gap is under two points. (The
synthetic smoke test had shown XGBoost at a +0.40 gap — on 40k generated rows. That was a
property of the stand-in, not of the problem, and it is a good example of why synthetic
results are never quoted.) Logistic regression's *negative* gap (validation above train)
is not by itself a bug. In expanding-window CV the training score includes the
small, early 2007–2012 vintages, while every validation fold comes from later periods. The
most likely reading is that the later vintages are simply easier to rank for a linear
model; I have not tested that.

**Tuning came last**, deliberately — after the comparison was settled, scored on PR-AUC
with `TimeSeriesSplit`. Tuning before comparing measures search effort, not model quality.
On the real data it bought **nothing**: tuned XGBoost 0.7162 vs default 0.7161 on test.
The defaults were already near the ceiling the features allow.

**Final out-of-time test results** (train 2007-06 → 2015-12, test 2016-01 → 2018-12):

| Model | ROC-AUC | PR-AUC | KS | Brier | ROC-AUC vs B1 | PR-AUC vs B1 |
|---|---|---|---|---|---|---|
| **XGBoost** | **0.7161** | **0.4068** | 0.314 | 0.1574 | **+0.0290** | **+0.0420** |
| XGBoost (tuned) | 0.7162 | 0.4067 | 0.313 | 0.1575 | +0.0291 | +0.0419 |
| HistGradientBoosting | 0.7148 | 0.4041 | 0.311 | 0.1574 | +0.0277 | +0.0393 |
| Logistic Regression | 0.7076 | 0.3895 | 0.301 | 0.1588 | +0.0205 | +0.0246 |
| Random Forest | 0.7065 | 0.3925 | 0.298 | 0.1586 | +0.0194 | +0.0277 |
| B1 `sub_grade` | 0.6871 | 0.3648 | 0.271 | 0.1741 | — | — |
| B0 majority | 0.5000 | 0.2242 | 0 | 0.1755 | −0.187 | −0.141 |

Three readings:

- **The honest band holds.** 0.7161 is inside 0.68–0.72 and inside the published range of
  0.678–0.735. No leakage alarm.
- **Every model beats the incumbent**, and the boosted models beat it most. The best model
  is ahead of logistic regression by only 0.0085 ROC-AUC (0.017 PR-AUC); the spread across
  all four models (0.0096) is a third of the lift over B1.
- **The lift is larger than the published ones** (+0.029 vs +0.012 to +0.018). Part of
  that is a different test window, but a lift above the published range is something to
  verify rather than celebrate: it has no confidence interval yet (Phase 16).

---

## Phase 11 — Calibration: the probability *is* the product

ROC-AUC is a pure ranking statistic. A model can rank perfectly and still be
systematically wrong about the *level* — and in credit risk the level is what gets used:
expected loss is `PD × EAD × LGD`, and pricing consumes the number itself.

Worse, I had made this problem deliberately: `class_weight="balanced"` and
`scale_pos_weight` distort the class prior to help the model learn, so raw outputs
**overstate** default probability.

**Decision:** isotonic calibration, fitted on a held-out slice of the **training** data —
specifically the *latest pre-cutoff months*, not a random subset, so the time ordering
holds throughout: fit on the early period, calibrate on the later one, test on the period
after that.

The calibration curve confirms it. For XGBoost, uncalibrated points sit well below the
diagonal (predicted ≫ observed). After isotonic calibration the Brier score improves from
**0.2050 to 0.1574**. ROC-AUC is unchanged — calibration is monotone and cannot alter
ranking. That invariance is exactly why AUC alone is insufficient.

The real run adds one wrinkle the synthetic one could not show. On the test period the
calibrated curve sits **slightly above** the diagonal: the model now *under*states default
by a few points. The calibration slice (2015-08 to 2015-12) learned a training-period
level, and the test period defaults at 22.4% against 18.4% in training (partly real,
partly the censoring in Phase 3). That is the case for the cheap recalibration trigger in
Phase 15.

---

## Phase 12 — The threshold is a policy decision, not a model output

At 20% prevalence, 0.5 is meaningless. A model can be well-calibrated and place nearly
every applicant below 0.5 — classifying everyone as "will repay", scoring 80% accuracy,
and being useless. (I demonstrate this numerically in notebook 01 rather than asserting
it, which is also why accuracy appears in my tables **only** to be dismissed.)

Three candidate rules — F1-optimal, Youden's J, and cost-based. Only the third encodes the
actual problem: a missed default costs the unrecovered principal; a wrongly declined
applicant costs the foregone margin. At `FN:FP = 4:1` the cut-off lands at **0.195–0.200**
for every model, far below 0.5, and the model correctly buys recall at the expense of
precision. On the test set, XGBoost at 0.200 catches **61.9%** of defaults at 36.5%
precision, declining about 38% of applicants, for an expected cost of 0.146 per loan
against 0.153 for B1 and 0.224 for approving everyone.

**Chosen on the calibration slice and applied unchanged to test.** Tuning a threshold on
test scores is the same family of leakage as fitting a scaler on them.

---

## Phase 13 — Evaluation: the aggregate number is the least interesting part

**Metrics I report and why** — PR-AUC primary (the rare class is the expensive one, and
its baseline is the prevalence, not 0.5); Brier co-primary (the only one that punishes
miscalibration); ROC-AUC, KS and Gini as the conventional credit-risk triplet; precision /
recall / F1 at the justified operating point; accuracy only as a cautionary exhibit.

**Error analysis.** An aggregate AUC hides everything operational. I sliced performance by
grade, purpose, term, FICO band and — most importantly — **issue quarter**. The time slice
asks the question the aggregate cannot: *is the model still working at the end of the test
window, or only at the start?* A rising FNR quarter by quarter is the expensive form of
drift. That slope is what sets the retraining cadence in Phase 15, derived rather than
guessed.

What the real slices show (XGBoost, test set):

| Slice | Finding |
|---|---|
| **Within grade** | ROC-AUC only **0.61–0.67** inside each grade. Most of the ranking power comes from separating grades, not from ordering applicants within one |
| **Term** | 60-month loans default at 33.4% vs 19.2%; the model declines 81% of them |
| **FICO band** | AUC rises from 0.68 (<680) to 0.77 (750+) — the model is weakest where risk is highest |
| **Issue quarter** | 0.742 in 2016Q1, then a flat 0.69–0.72 through 2018Q2; **2018Q4 collapses to 0.551** on 5,018 loans with a 2.4% default rate — the censoring artefact from Phase 3, not model failure |

**Fairness.** Lending decisions affect people, so I measured decline rate, FPR and FNR
across US region, income band and housing status, with a disparate-impact ratio against
the 0.8 convention.

The interpretation that matters: a group with a higher decline rate **and** a
correspondingly higher realised default rate is being treated consistently. A group with a
higher decline rate **but a similar default rate and elevated FPR** is being penalised by
the model rather than by its own risk. Decline rates alone cannot separate the two.

The real results (XGBoost at the cost threshold):

| Grouping | Disparate-impact ratio | Verdict | Reading |
|---|---|---|---|
| US region | 0.850 | pass | Decline rates 35–41%, tracking default rates 21–24% |
| **Income band** | **0.534** | **flag** | Lowest quartile: 49.2% declined vs 26.2% for the highest. Default rates differ by 1.43× (26.4% vs 18.4%), decline rates by 1.88×, and **FPR doubles** (0.418 vs 0.209) |
| **Housing** | **0.674** | **flag** | Renters: 46.5% declined vs 31.4% for mortgage holders; default 27.2% vs 18.7%; FPR 0.383 vs 0.259 |

Both flags are **partly risk-consistent and partly not**. Low-income applicants and renters
really do default more, but the decline gap is wider than the default gap, and the
doubled FPR means more creditworthy low-income applicants are wrongly declined. That is
precisely the pattern the interpretation rule above was written to catch. It is a finding
to investigate (for example, whether `annual_inc` and `loan_to_income` are carrying more
weight than their risk content), not a verdict.

**The caveat I keep attached:** this dataset contains **no protected attributes**. Region,
income and housing are proxies. A disparity found here is real and worth investigating;
an absence of disparity does **not** certify fairness on the attributes that matter
legally. This is a screen, not a compliance audit — and none of the comparable published
projects does even this much.

**The optimism measurement.** I re-ran everything once under a random split. Every model
scored higher on ROC-AUC, by **+0.008 to +0.012** — almost exactly the one point Xia et al.
measured between out-of-sample and out-of-time on this dataset. That gap is the size of the
illusion in the published leaderboards, and it is why a shared split protocol is not
bureaucracy.

PR-AUC went the *other* way (OOT 0.4068 vs random 0.3962 for XGBoost), which `CLAUDE.md`
says to investigate rather than report. The cause is prevalence, not leakage: PR-AUC's
floor is the positive rate, which is 22.4% on the OOT test set and 20.0% on the random
one. Relative to its floor, OOT is lower as expected — 1.81× prevalence vs 1.98×.

---

## Phase 14 — Explanation, and one result I do not trust

Permutation importance (what the model relies on), logistic coefficients (direction and
strength), SHAP (per-applicant attribution — the question an adverse-action notice legally
has to answer), partial dependence (the shape of each effect, and whether it is monotone).

On the real data **`sub_grade_ordinal` dominates** permutation importance: shuffling it
costs 0.037 ROC-AUC, three times more than the next column (`term`, 0.012), followed by
`int_rate`, `revol_bal` and `grade_ordinal`. The logistic coefficients agree on
`sub_grade_ordinal` first, then `purpose = small_business`. Several `addr_state` dummies
also carry sizeable coefficients, which matters for the region screen in Phase 13. My
reading in the notebook is that the model is substantially **re-learning Lending Club's
own underwriting** rather than adding information.

**But I should flag that this claim is untested, and the one published test points the
other way.** One surveyed project removed `grade`/`sub_grade` entirely and AUC moved
0.7350 → 0.7335 — essentially nothing (`docs/related_work.md`). My interpretation is
plausible and I have not earned it. The ablation is cheap and is on the action list.

The general rule I hold here: partial dependence and SHAP describe **the model**, not the
world. If a direction contradicts underwriting knowledge, suspect leakage or a data defect
before announcing a discovery.

---

## Phase 15 — Shipping it, and knowing when it has died

**Packaging.** I save the *entire* `Pipeline` — preprocessing, estimator, calibration — as
one joblib artifact. Saving the estimator alone is the classic deployment bug: production
then re-implements the preprocessing, and any discrepancy silently corrupts every
prediction.

It is refitted on the full training period (including the calibration slice — legitimate,
that was always training data). **The test period is still never touched**, which is what
keeps its metrics an honest estimate.

I assert a **round-trip**: reload from disk and confirm predictions are bit-identical.
Writing a file is not the same as writing a usable file, and this is the check that catches
an unpicklable transformer or a version mismatch here rather than in production.

A model card records the training window, feature list, metrics, split hash and library
versions.

**Which model gets packaged.** Notebook 04 ships **logistic regression + isotonic
calibration** (OOT ROC-AUC 0.7076, PR-AUC 0.3895). That choice was originally justified by
the synthetic run, where the ensembles overfit. On real data they do not, and XGBoost is
ahead by 0.0085 ROC-AUC / 0.017 PR-AUC. Logistic regression is still defensible —
coefficient-level auditability is what an adverse-action notice needs — but the reason is
now governance, not generalisation, and it costs measurable performance. **Decision: keep
logistic regression**, accepting the 0.0085 ROC-AUC cost in exchange for a model whose
every decline can be explained coefficient by coefficient. The project's headline is
therefore the packaged model's lift over B1, **+0.0205 ROC-AUC / +0.0246 PR-AUC**; XGBoost's
+0.029 is reported as the ceiling the features allow, not as the deployed result.

On the real run the round-trip check passed bit-for-bit (max difference 0.0).

**Monitoring.** Two different failure modes need two different instruments:

- **Data drift** — the inputs move. Measured with **PSI** per feature against the training
  baseline (0.1 investigate / 0.25 retrain). Crucially, **score PSI needs no labels**, so
  it is available the day applications are scored, whereas outcomes take years. It is the
  earliest warning that exists.
- **Concept drift** — the inputs look the same but their relationship to default changed.
  PSI is blind to this. Only realised outcomes reveal it, which is why the quarterly
  performance-decay curve exists.

The retraining policy ties numeric triggers to actions, with one operational distinction
worth stating: **recalibration is not retraining**. If ranking still holds but the
probabilities have drifted, refitting only the isotonic layer is far cheaper and lower
risk.

**What the real run measured:**

- **Score PSI 0.0064** — stable. Every feature PSI is below 0.1; the largest are
  `revol_util` (0.098) and `int_rate` (0.083), both close to the "investigate" line.
- **Performance decay.** The notebook fits −0.008 ROC-AUC per quarter across the test
  window and prints "refit more often than annually". **That slope is an artefact.** It is
  driven by the two unmatured quarters (2018Q3 0.683, 2018Q4 0.531 on 5,018 loans). Drop
  them, as the notebook's own text says to, and the slope is **−0.001 per quarter**
  (about −0.004 per year). The honest reading: the inputs are stable, ranking holds through
  2018Q2, and the level is drifting (Phase 11) — a **recalibration** case more than a
  retraining one. The notebook code does not yet exclude those quarters automatically.

---

## Phase 16 — What is still wrong with this

The honest section. Each item names work that does better.

**1. The headline lift is unverified.** The packaged logistic regression beats B1 by
+0.0205 ROC-AUC, just above the published range (+0.012 to +0.018); the best model,
XGBoost, by +0.029, well above it. It might be real — the test window and feature set differ — but
a lift above the published range is a red flag until a confidence interval (item 6) and
the grade ablation (Phase 14) say otherwise.

**2. I documented right-censoring; others solved it.** The real run shows the damage
concretely: only 37.8% of post-cutoff loans had resolved, the split ratio flipped to 61/39,
2016–2017 test quarters are enriched with defaults, 2018 quarters with prepayments, and
the fake decay slope in Phase 15 comes from those last quarters. One surveyed project
restricts to 36-month loans issued 2012–2015, all of which had matured by the 2018 Q4
snapshot — leaving 0.025% unresolved. That is strictly better than my approach of
filtering and writing a limitation. It is still the highest-value change available, and
it would also unlock the 49 `sparse_pre2012_bureau` columns.

**3. The packaged model is not the best model — by choice.** Logistic regression was kept
on governance grounds (Phase 15), giving up 0.0085 ROC-AUC to XGBoost. The decision is
recorded; the cost is real and should be restated whenever the model is presented.

**4. Two fairness flags are open.** Income band (disparate-impact ratio 0.534) and
housing (0.674) fail the 0.8 convention, with FPR roughly doubled for the lowest income
quartile (Phase 13). Screened, not explained.

**5. My cost ratio is invented.** `4:1` is a placeholder. A surveyed project *derived* the
real interest margin from the data and found the optimal threshold **doubled** (0.25 →
0.50) versus their placeholder. Every threshold-dependent number I report rests on a guess
that is demonstrably load-bearing.

**6. No confidence interval on the lift.** I report +0.0205 (packaged) and +0.0290 (best) over B1 as point estimates.
Published work reports paired-bootstrap CIs. Without one I cannot claim the lift is
distinguishable from zero.

**7. Leakage rules are documented, not enforced.** Mine is a written list plus a manual
audit. Two surveyed projects **fail the training run** automatically if a banned column
appears. Theirs survives a careless future edit; mine depends on the next person reading
the docs.

**8. `issue_d` is a proxy for application date.** It is the *issue* date; the dataset never
records when the decision was made. Any application-to-issuance lag is invisible, so my
features are dated slightly later than a live scorecard would see them. Standard practice
on this dataset, but it should be stated, not assumed.

**9. Accepted applicants only.** The model estimates risk *conditional on acceptance*, not
for the through-the-door population. This is the reject-inference problem and it cannot be
fixed by joining the rejected-applications file, which has no outcomes by construction.

---

## The decision log, in one table

| # | Decision | Alternative rejected | Why |
|---|---|---|---|
| 1 | Baselines fixed before modelling | Compare models to each other | Lift over the incumbent is the only claim that matters |
| 2 | `usecols` whitelist + defensive drop | Drop leakage columns after loading | Two layers; the error is silent and fatal |
| 3 | Terminal statuses only | Treat `Current` as repaid | Would label unresolved loans as successes |
| 4 | Out-of-time split | Random stratified | Deployment scores future applicants |
| 5 | Cutoff date, not seed | `random_state=42` | Reproducible across implementations |
| 6 | Kept the 61/39 ratio the date produces | Move the cutoff to reach 80/20 | Choosing a split by its outcome |
| 7 | Fixed domain clipping | 1st/99th percentile | A percentile is learned from data |
| 8 | Missing indicators | `fillna(median)` | The blank is protective information |
| 9 | Train-only column screening | Screen on the full frame | A missing rate is a statistic |
| 10 | Ordinal grade encoding | One-hot | Discards a monotone ordering |
| 11 | No target encoding | WOE / mean encoding on `addr_state` | Each row would see its own answer |
| 12 | Everything learned inside a `Pipeline` | Transform then split | Makes contamination structurally impossible |
| 13 | `TimeSeriesSplit` for CV | `StratifiedKFold` | Shuffled folds restore the look-ahead |
| 14 | Isotonic calibration on late training months | No calibration / random slice | The probability is the product |
| 15 | Cost-based threshold | 0.5 | 0.5 is meaningless at 20% prevalence |
| 16 | Reported the lift over B1 as the headline, and flagged it as above the published range | Headline the absolute AUC | The incumbent is the bar; an unusually large win is a claim to verify |
| 17 | Save the whole pipeline | Save the estimator | Production must not re-implement preprocessing |
| 18 | Score PSI as primary monitor | Wait for outcomes | Labels arrive years late |

---

## If you read only one thing

The decisions that most affected the final numbers were **not** the algorithm. In order of
impact: whether the split respects time, whether the preprocessing respects the split,
where the decision threshold sits, and whether the model is compared against the rule it
is meant to replace. Model choice came a distant fifth: on the real file all four models
sit within 0.0096 ROC-AUC of each other, a third of the 0.029 lift over the incumbent.
Boosting wins, narrowly — and whether that margin is worth the interpretability is a
governance call, not a modelling one.

**References:** benchmark comparison with citations in `docs/related_work.md`; rules and
conventions in `CLAUDE.md`; the implementation in `notebooks/01`–`04`.
