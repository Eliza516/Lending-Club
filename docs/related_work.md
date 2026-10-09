# Related Work — How This Project Compares

Required by the course brief: *"Chứng minh được cái mình làm hơn gì những cái đã có trên
mạng Kaggle, Github, blogs."*

Every number below is quoted from the cited source. Sources were retrieved in September
2026; see the Sources section for URLs.

---

## 1. The headline caveat, stated first

**This project's current numbers come from a synthetic sample, not the real Kaggle file.**
They smoke-test the pipeline. They are **not comparable** to any number in this document,
and must not be quoted as results.

| | ROC-AUC | Status |
|---|---|---|
| Our Logistic Regression (synthetic) | 0.6421 | **Not a result** — synthetic data |
| Our B1 `sub_grade` incumbent (synthetic) | 0.6164 | **Not a result** — synthetic data |
| Our lift over incumbent (synthetic) | +0.0257 | **Not a result** — synthetic data |

So the honest claim right now is **methodological, not numerical**. Section 4 lists where
we are genuinely ahead; section 5 lists where we are genuinely behind. Both matter.

---

## 2. Published benchmarks on this dataset

| Project | Split | Best model AUC | Incumbent baseline | Lift |
|---|---|---|---|---|
| **vaibhavkev/credit-risk** | **OOT**: train Jan 2012–Jun 2014, val Jul–Dec 2014, test 2015 | XGBoost (monotone) **0.696** (Gini 0.392, KS 0.284); LogReg 0.683 | LC sub-grade **0.679**, LC rate 0.678 | **+0.018**, paired-bootstrap 95 % CI [+0.016, +0.020] |
| **Arturo-GA/lendingclub-default-risk** | **OOT**: 70 % oldest train / 15 % val / 15 % newest test | LightGBM **0.7151** (Gini 0.4302, KS 0.3137); LogReg 0.7041; WoE scorecard 0.704 | not reported | — |
| **Tanish-Srivastava/credit-risk-scorecard** | random | WoE + LogReg **0.692** (Gini 0.385, KS 27.7) | LC grade **0.680** | +0.012 |
| **Arnav618/lending-club-credit-risk** | not specified | tuned XGBoost **0.7265** (baseline 0.7177), bootstrap CI [0.7127, 0.7226] | not reported | — |
| **Nasha14/Lending-Club-Loan-Default-Prediction** | not specified | XGBoost **0.735**; LogReg 0.649 | not reported | — |
| **Xia et al. (2020), SurvXGBoost**, peer-reviewed | out-of-sample vs out-of-time | out-of-sample **68.07 %**, **out-of-time 67.07 %** | — | — |

### 2.1 Two findings that validate our design

**Out-of-time splits score lower than random ones — confirmed independently.** The two
OOT projects land at 0.696 and 0.7151; the random-split and unspecified-split projects
report 0.692, 0.7265 and 0.735. Xia et al. measure the gap directly within one study:
**68.07 % out-of-sample vs 67.07 % out-of-time, a drop of about one point.**

This is exactly what our notebook 03 §9.8 argues and measures. It also means **leaderboard
comparisons across these projects are not valid** unless the split protocol matches — which
is precisely why the course brief requires groups on this dataset to share one split.

**Our documented "honest band" is right.** `CLAUDE.md` states ROC-AUC ≈ 0.68–0.72 is the
realistic range and anything near 0.99 indicates leakage. The published range across six
independent projects is **0.678 – 0.735**. The band holds.

### 2.2 The incumbent is hard to beat — confirmed

Both projects that bothered to benchmark against Lending Club's own grade found the margin
is **small**:

- vaibhavkev: model 0.696 vs LC sub-grade 0.679 → **+0.018**
- Tanish-Srivastava: model 0.692 vs LC grade 0.680 → **+0.012**, described as
  *"comparable to, and marginally better than"* the platform's 7-level rating

This strongly supports the B1 baseline design in our notebook 01 §1.4 / 03 §8.2. A project
reporting a large lift over `sub_grade` should be treated as suspect.

---

## 3. Leakage: the field-wide failure mode

Our leakage discipline is not unusual among *good* projects — it is the dividing line
between serious and unserious ones.

- **Arnav618** caught it empirically: `total_rec_prncp` scored mutual information
  *"0.47 — more than 10x higher than any other feature"*, and after removing
  *"30+ leakage columns"* the top legitimate feature scored *"0.04"*.
- **Arturo-GA** documents *"~40 columnas que describen la vida del préstamo"* and
  `features.assert_no_leakage` **fails the training run** if a banned column appears.
- **vaibhavkev** ships `tests/test_leakage.py`, which *"fails if a column is unclassified
  or anything other than application-time data reaches the model."*

Our project blocks the same columns and audits them, but see §5.1 — two of these enforce it
in code and we do not.

---

## 4. Where this project is genuinely ahead

Claims here are about **method**, verified against what each source documents.

### 4.1 Fairness screening — the clearest gap

**None of the six Lending Club projects surveyed performs a fairness analysis.**

- Arturo-GA: *"Neither fairness metrics nor continuous monitoring were explicitly
  addressed."*
- Arnav618: fairness *"not addressed."*
- Nasha14: *"not mentioned."*
- vaibhavkev takes a different route — it **removes** geography structurally
  (*"zip_code and addr_state are dropped; location is a well-known proxy for protected
  characteristics"*) — a defensible design choice, but it *avoids* the question rather than
  measuring it.

Our notebook 03 §9.7 computes decline rate, FPR, FNR and a disparate-impact ratio across
region, income band and housing status. **Caveat we keep prominent:** these are proxies,
the dataset has no protected attributes, so it is a screen, not a compliance audit.

### 4.2 Problem framing as an explicit artifact

No surveyed project has an equivalent of our notebook 01: the four framing questions, the
cost matrix, the "do we even need ML?" test, and the baselines **specified before any
result is seen**. Most projects open directly with data loading.

### 4.3 Overfitting diagnosis via a train-vs-CV gap table

Not reported by any surveyed project. In our run it is what explains the model ranking —
XGBoost gap +0.396 and HistGB +0.332 versus logistic regression +0.044.

### 4.4 Missingness treated as signal

Our §6.3 separates *"missing because it does not exist"* from *"missing because it was not
recorded"* and adds `_was_missing` indicators. `mths_since_last_delinq` being blank means a
clean record — median imputation destroys that. No surveyed project documents this.

### 4.5 Breadth in one place

Individually, most of our components exist somewhere: calibration (Tanish-Srivastava),
PSI (Arturo-GA), cost-based thresholds (Arnav618), OOT + incumbent benchmark
(vaibhavkev). **No single surveyed project has all of them**, and none adds fairness.

---

## 5. Where this project is genuinely behind

This section is not padding. These are real deficits against specific published work.

### 5.1 Right-censoring: solved by others, only documented by us

Our limitation 2 says the terminal-status filter enriches the late test period with loans
that resolved early. **vaibhavkev actually fixes this**: restrict to *"36-month loans
issued 2012–2015"* which *"had all reached maturity by the 2018 Q4 snapshot"*, leaving only
*"147 of 589,635 (0.025 %)"* unresolved.

That is a strictly better design than ours. Adopting it is the single highest-value change
available to this project.

### 5.2 Our cost ratio is invented; Arnav618's is measured

We use a placeholder `FN:FP = 4:1`. Arnav618 **derived** the real interest margin from the
data — *"(installment × term) − loan amount, averaged across all repaid loans"* — getting
27.2 % versus a 10 % placeholder, which *"shift[ed] the optimal decision threshold from
0.25 to 0.50"*. The threshold **doubled**. Our headline threshold-dependent numbers rest on
a guess that is demonstrably load-bearing.

### 5.3 No confidence interval on our lift

vaibhavkev reports +0.018 AUC with a **paired-bootstrap 95 % CI [+0.016, +0.020]**; Arnav618
reports a bootstrap CI too. We report a point estimate. Without a CI we cannot claim our
lift over B1 is statistically distinguishable from zero.

### 5.4 No business translation

vaibhavkev converts model advantage into money: at a 5 % decline rate the model caught
*"5,192"* defaults versus Lending Club's *"4,705"*, worth *"+$6.1M"* net, rising to
*"+$16.7M"* at a 20 % decline rate. Far more persuasive to a credit committee than an AUC
delta.

### 5.5 Leakage rules are documented, not enforced

Ours is a documented list plus manual audit. Arturo-GA and vaibhavkev both **fail the run**
automatically. Theirs survives a careless future edit; ours depends on the next person
reading `CLAUDE.md`.

### 5.6 Untested claim about re-learning the incumbent

Our notebooks assert that heavy reliance on `int_rate`/`sub_grade` means the model is
largely reproducing Lending Club's underwriting. **Nasha14 tested this and found the
opposite**: removing `grade`/`sub_grade` changed AUC only *"0.7350 → 0.7335"* — essentially
nothing. Our claim is plausible but currently unverified, and the one published test of it
points the other way.

### 5.7 No results on real data yet

The others ran on ~1.3M resolved loans. We have not.

---

## 6. Priority actions

| # | Action | Evidence |
|---|---|---|
| 1 | Run the pipeline on the real Kaggle file | everything else is blocked on this |
| 2 | Adopt matured-vintage selection to fix right-censoring | §5.1, vaibhavkev |
| 3 | Derive the cost ratio from interest margin instead of guessing | §5.2, Arnav618 |
| 4 | Add paired-bootstrap CI to the lift over B1 | §5.3, vaibhavkev |
| 5 | Run the grade/sub_grade ablation to test our own claim | §5.6, Nasha14 |
| 6 | Convert the lift into dollars at several decline rates | §5.4, vaibhavkev |
| 7 | Make the leakage check fail the run, not just document it | §5.5, Arturo-GA |

Items 2–7 are all cheap. Together they would move this project from "comparable method,
untested numbers" to a defensible result.

---

## Sources

Retrieved September 2026.

1. vaibhavkev, *credit-risk* — https://github.com/vaibhavkev/credit-risk
2. Arturo-GA, *lendingclub-default-risk* — https://github.com/Arturo-GA/lendingclub-default-risk
3. Tanish-Srivastava, *credit-risk-scorecard* — https://github.com/Tanish-Srivastava/credit-risk-scorecard
4. Arnav618, *lending-club-credit-risk* — https://github.com/Arnav618/lending-club-credit-risk
5. Nasha14, *Lending-Club-Loan-Default-Prediction* — https://github.com/Nasha14/Lending-Club-Loan-Default-Prediction
6. yanxiali, *Predicting-Default-Clients-of-Lending-Club-Loans* — https://github.com/yanxiali/Predicting-Default-Clients-of-Lending-Club-Loans
7. Xia, Y., He, L., et al. (2020). *A dynamic credit scoring model based on survival gradient boosting decision tree approach.* Technological and Economic Development of Economy. https://journals.vilniustech.lt/index.php/TEDE/article/view/13997
8. *Machine learning powered financial credit scoring: a systematic literature review.* Artificial Intelligence Review (2025). https://link.springer.com/article/10.1007/s10462-025-11416-2
9. kozodoi, *Fair_Credit_Scoring* — https://github.com/kozodoi/Fair_Credit_Scoring

**Note on citation practice.** The brief requires every reference to be cited in the body,
not only listed. Each source above is cited at the point its number is used.
