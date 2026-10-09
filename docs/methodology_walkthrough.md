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

The raw file is ~2.26M rows × 151 columns, 2007–2018 Q4. (Shape and the resolved-row
counts below are as documented for the real Kaggle file and corroborated by the projects
surveyed in `docs/related_work.md` — this project has not yet run on it; see Phase 16.)
Three things I check before anything else, in this order.

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

A 36-month loan issued in late 2018 cannot be `Fully Paid` by the 2018 Q4 snapshot. It can
only be resolved if it resolved **early** — and early resolution is disproportionately
*charge-off*. So filtering to terminal statuses silently enriches the recent period with
defaults.

**Consequence:** the observed default rate in the late period is **not** the true default
rate. I can see this directly in the default-rate-over-time plot, which turns upward at
the right-hand edge. Some of that is real drift; some is this artefact. I cannot cleanly
separate them.

**Decision:** keep the filter, document the bias as a named limitation, and never quote
the test-period default rate as a cohort rate. (Phase 16 revisits this — it is the one
place where a published project does better than me.)

On the real file this leaves ~1.3M resolved loans at a ~20% default rate, consistent
across the surveyed projects.

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
- **The resulting ratio is ~50/50, not 80/20.** Volume growth makes 2016 the middle of the
  resolved data. I did *not* move the cutoff to hit a nicer ratio — choosing a split by
  looking at what it produces is choosing by outcome.
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
> row set.

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

**(c) Outliers — and the trap.** The obvious move is to clip at the 1st/99th percentile.
**That is leakage.** A percentile is a statistic computed from the data; using one before
the split lets the test period's income distribution shape how training rows are
transformed. I clip at **fixed domain bounds** instead (`annual_inc` ≤ $1.5M, `dti` ≤ 60)
— limits that come from underwriting plausibility, not from where a quantile happens to
fall.

The same logic forced `revol_util` buckets to fixed edges (0/25/50/75/100) rather than
quantile bins.

**(d) Label normalisation.** `home_ownership` carries `ANY`, `NONE` and `OTHER` — three
labels for one residual category, an artefact of the application form changing over the
years. Left alone they become three sparse dummies meaning the same thing, and worse,
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

I verified it was worth doing rather than assuming: on training rows, applicants with
`mths_since_last_delinq` missing default at **16.2%** versus **18.1%** for those with a
value. The blank is protective, exactly as the domain reasoning predicted. If the two
rates had matched I would have dropped the indicator — a column that adds no information
is just variance.

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
(a) and differ from (b) — maximum gap 275 units on one column. If preprocessing had been
contaminated, they would have matched (b).

---

## Phase 9 — Baselines first

Before any model:

| Baseline | Test ROC-AUC | Reading |
|---|---|---|
| B0 majority class | 0.5000, PR-AUC = prevalence | The floor. Beating it proves nothing |
| **B1 `sub_grade` (incumbent)** | **0.6164** | The real bar |

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

I asked for train-fold scores alongside validation scores, and that table is where the
project's most useful finding came from:

| Model | Train AUC | CV AUC | Gap |
|---|---|---|---|
| Logistic Regression | 0.675 | 0.632 | **+0.044** |
| Random Forest | 0.785 | 0.644 | +0.140 |
| HistGradientBoosting | 0.938 | 0.606 | **+0.332** |
| XGBoost | 0.992 | 0.596 | **+0.396** |

The ensembles are memorising. XGBoost fits the training folds almost perfectly and
generalises worst. Without this column the final ranking would look arbitrary; with it,
the result is obvious and explainable.

**Tuning came last**, deliberately — after the comparison was settled, scored on PR-AUC
with `TimeSeriesSplit`. Tuning before comparing measures search effort, not model quality.

**Final test results (synthetic sample — see the caveat below):**

| Model | ROC-AUC | PR-AUC | vs B1 |
|---|---|---|---|
| Logistic Regression | 0.6421 | 0.2920 | **+0.0257** |
| Random Forest | 0.6387 | 0.2834 | +0.0223 |
| B1 `sub_grade` | 0.6164 | 0.2692 | — |
| HistGradientBoosting | 0.6279 | 0.2680 | +0.0115 |
| XGBoost | 0.6037 | 0.2494 | **−0.0127** |

Two ensembles land **below the incumbent**. I report that rather than tuning until it
disappears. The interpretable model winning is the honest outcome here, and it is also the
one a regulator can be shown.

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

The calibration curve confirms it: uncalibrated points sit well below the diagonal
(predicted ≫ observed); after isotonic they sit on it, and the Brier score improves from
0.2315 to 0.1477. ROC-AUC is unchanged — calibration is monotone and cannot alter ranking.
That invariance is exactly why AUC alone is insufficient.

---

## Phase 12 — The threshold is a policy decision, not a model output

At 20% prevalence, 0.5 is meaningless. A model can be well-calibrated and place nearly
every applicant below 0.5 — classifying everyone as "will repay", scoring 80% accuracy,
and being useless. (I demonstrate this numerically in notebook 01 rather than asserting
it, which is also why accuracy appears in my tables **only** to be dismissed.)

Three candidate rules — F1-optimal, Youden's J, and cost-based. Only the third encodes the
actual problem: a missed default costs the unrecovered principal; a wrongly declined
applicant costs the foregone margin. At `FN:FP = 4:1` the cut-off lands near 0.195, far
below 0.5, and the model correctly buys recall at the expense of precision.

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

**Fairness.** Lending decisions affect people, so I measured decline rate, FPR and FNR
across US region, income band and housing status, with a disparate-impact ratio against
the 0.8 convention.

The interpretation that matters: a group with a higher decline rate **and** a
correspondingly higher realised default rate is being treated consistently. A group with a
higher decline rate **but a similar default rate and elevated FPR** is being penalised by
the model rather than by its own risk. Decline rates alone cannot separate the two.

**The caveat I keep attached:** this dataset contains **no protected attributes**. Region,
income and housing are proxies. A disparity found here is real and worth investigating;
an absence of disparity does **not** certify fairness on the attributes that matter
legally. This is a screen, not a compliance audit — and none of the comparable published
projects does even this much.

**The optimism measurement.** I re-ran everything once under a random split. Every model
scored higher. That gap is the size of the illusion in the published leaderboards for this
dataset, and it is why a shared split protocol is not bureaucracy.

---

## Phase 14 — Explanation, and one result I do not trust

Permutation importance (what the model relies on), logistic coefficients (direction and
strength), SHAP (per-applicant attribution — the question an adverse-action notice legally
has to answer), partial dependence (the shape of each effect, and whether it is monotone).

`int_rate` and `sub_grade` dominate. My reading in the notebook is that the model is
substantially **re-learning Lending Club's own underwriting** rather than adding
information.

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

---

## Phase 16 — What is still wrong with this

The honest section. Each item names work that does better.

**1. The results are synthetic.** The numbers above come from a generated stand-in,
because the 1.6GB Kaggle file is not in the environment. They smoke-test the pipeline and
are **not findings**. Everything else here is blocked on running the real file.

**2. I documented right-censoring; others solved it.** One surveyed project restricts to
36-month loans issued 2012–2015, all of which had matured by the 2018 Q4 snapshot —
leaving 0.025% unresolved. That is strictly better than my approach of filtering and
writing a limitation. It is the highest-value change available.

**3. My cost ratio is invented.** `4:1` is a placeholder. A surveyed project *derived* the
real interest margin from the data and found the optimal threshold **doubled** (0.25 →
0.50) versus their placeholder. Every threshold-dependent number I report rests on a guess
that is demonstrably load-bearing.

**4. No confidence interval on the lift.** I report +0.0257 over B1 as a point estimate.
Published work reports paired-bootstrap CIs. Without one I cannot claim the lift is
distinguishable from zero.

**5. Leakage rules are documented, not enforced.** Mine is a written list plus a manual
audit. Two surveyed projects **fail the training run** automatically if a banned column
appears. Theirs survives a careless future edit; mine depends on the next person reading
the docs.

**6. `issue_d` is a proxy for application date.** It is the *issue* date; the dataset never
records when the decision was made. Any application-to-issuance lag is invisible, so my
features are dated slightly later than a live scorecard would see them. Standard practice
on this dataset, but it should be stated, not assumed.

**7. Accepted applicants only.** The model estimates risk *conditional on acceptance*, not
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
| 6 | Kept the ~50/50 ratio | Move the cutoff to reach 80/20 | Choosing a split by its outcome |
| 7 | Fixed domain clipping | 1st/99th percentile | A percentile is learned from data |
| 8 | Missing indicators | `fillna(median)` | The blank is protective information |
| 9 | Train-only column screening | Screen on the full frame | A missing rate is a statistic |
| 10 | Ordinal grade encoding | One-hot | Discards a monotone ordering |
| 11 | No target encoding | WOE / mean encoding on `addr_state` | Each row would see its own answer |
| 12 | Everything learned inside a `Pipeline` | Transform then split | Makes contamination structurally impossible |
| 13 | `TimeSeriesSplit` for CV | `StratifiedKFold` | Shuffled folds restore the look-ahead |
| 14 | Isotonic calibration on late training months | No calibration / random slice | The probability is the product |
| 15 | Cost-based threshold | 0.5 | 0.5 is meaningless at 20% prevalence |
| 16 | Reported ensembles losing to B1 | Tune until they win | It is the result |
| 17 | Save the whole pipeline | Save the estimator | Production must not re-implement preprocessing |
| 18 | Score PSI as primary monitor | Wait for outcomes | Labels arrive years late |

---

## If you read only one thing

The decisions that most affected the final numbers were **not** the algorithm. In order of
impact: whether the split respects time, whether the preprocessing respects the split,
where the decision threshold sits, and whether the model is compared against the rule it
is meant to replace. Model choice came a distant fifth — and the simplest model won.

**References:** benchmark comparison with citations in `docs/related_work.md`; rules and
conventions in `CLAUDE.md`; the implementation in `notebooks/01`–`04`.
