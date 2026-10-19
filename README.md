# Explainable AI & Fairness Analysis — Credit Card Default Prediction

**Explainable ML · Model evaluation · Fairness assessment**

A documented AI Ethics course case study comparing Random Forest and Explainable Boosting Machine models for credit-card default prediction.

> **Repository status:** The original [Jupyter notebook](explainable_fair_credit.ipynb) and saved SHAP plots are included. The dataset is not included; a fresh execution has not been verified.

This project applies Explainable AI and fairness auditing techniques to a credit card default prediction task. Using the UCI Credit Card Default dataset, it trains machine learning models to predict whether a client will default on their next payment, then interrogates those models with a suite of interpretability tools and evaluates them for bias across gender, age, and marital status.

## 📊 Dataset

**[Default of Credit Card Clients (UCI / Taiwan)](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)** — 30,000 clients with demographic attributes (sex, age, education, marriage), credit limits, payment history, bill statements, and payment amounts.

- **Target:** `default.payment.next.month` (1 = default, 0 = no default)
- Class imbalance in the training set is handled with **SMOTE** oversampling.


## 🧠 Models

| Model | Library | Purpose |
|---|---|---|
| Random Forest Classifier | scikit-learn | Black-box baseline explained post-hoc |
| Explainable Boosting Machine (EBM) | InterpretML | Inherently interpretable glass-box model |

Both models are trained on SMOTE-balanced data and evaluated with classification reports (precision, recall, F1) on a held-out test set.

## 🔍 Explainability Techniques

### Global explanations
- **EBM global feature importance** — built-in interpretability from the glass-box model
- **Random Forest feature importances** — impurity-based ranking
- **SHAP (TreeExplainer)** — summary plots showing feature impact across the test set

### Local explanations
- **SHAP waterfall plots** — per-instance breakdowns of how each feature pushes a prediction toward default or non-default
- **EBM local explanations** — instance-level views for both positive and negative examples
- **LIME** — local surrogate explanations for individual predictions

### Rule-based & counterfactual explanations
- **Anchors (Alibi)** — high-precision IF-THEN rules that "anchor" a prediction, with precision and coverage reported
- **Counterfactuals (DiCE)** — minimal feature changes that would flip a prediction (default → no default, and vice versa)

## ⚖️ Fairness Evaluation

The analysis compares model outcomes across three sensitive attributes using two industry-standard toolkits:

**Fairlearn** — per-group accuracy, selection rate, true/false positive rates, plus:
- Demographic (statistical) parity difference
- Equalized odds difference

**AIF360** — disparate impact, statistical parity difference, equal opportunity difference, and average odds difference for:
- **Gender** (Male vs. Female)
- **Age groups** (Young < 30, Middle-aged 30–50, Older > 50 — pairwise comparisons)
- **Marital status** (Married vs. Single)

## 📈 Results

### Data preparation
- The dataset has **30,000 clients** and **no missing values**.
- The target is imbalanced, so **SMOTE** was used to balance the training set to **16,355 samples per class** (default vs. no-default) before training.

### Model performance

Both models were evaluated on a held-out test set. The Random Forest slightly outperforms the EBM overall, but both struggle to recall the minority "default" class — a common and important pattern in credit risk data.

| Model | Accuracy | Default (class 1) Precision | Default Recall | Default F1 |
|---|---|---|---|---|
| **Random Forest** | 0.77 | 0.49 | 0.48 | 0.48 |
| **EBM (glass-box)** | 0.75 | 0.44 | 0.47 | 0.46 |

Both models predict the "no default" class well (F1 ≈ 0.85) but only catch roughly **half of actual defaulters**. In a lending context this is the costly type of error, and it's exactly why the explainability and fairness analysis below matters.

### Global feature importance (Random Forest)

The most influential features for predicting default were the **repayment status and financial-history variables**, not the demographic ones:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `PAY_0` (most recent repayment status) | 0.100 |
| 2 | `BILL_AMT1` (latest bill amount) | 0.096 |
| 3 | `LIMIT_BAL` (credit limit) | 0.095 |
| 4 | `PAY_AMT1` (latest payment amount) | 0.086 |
| … | … | … |
| 12 | `EDUCATION` | 0.036 |
| 13 | `MARRIAGE` | 0.035 |
| 14 | `SEX` | **0.017** (lowest) |

In the reported ranking, `SEX` has the lowest impurity-based importance. Low importance does not establish fairness: correlated features can carry demographic information, and group-level outcomes require separate assessment.

### SHAP global explanation

![SHAP summary plot](images/shap_summary.png)

The SHAP beeswarm summary plot confirms the feature-importance ranking and adds *direction*. Each dot is one client; red = high feature value, blue = low. `PAY_0` dominates: **high recent-repayment-delay values (red) push predictions strongly toward default** (positive SHAP), while clients who paid on time (blue) are pushed toward no-default. Bill and payment amounts contribute in a more mixed way, and demographic features cluster near zero impact.

### SHAP local explanations (individual predictions)

Waterfall plots break down a single prediction, showing how each feature moved it from the base rate to the final output.

**Instance 0**
![SHAP waterfall for instance 0](images/shap_waterfall_instance0.png)

**Instance 1**
![SHAP waterfall for instance 1](images/shap_waterfall_instance1.png)

These per-client breakdowns are what make the model *actionable*: for any individual decision you can see exactly which features (usually recent repayment status and bill/payment amounts) pushed the prediction up or down.

### Anchor explanations (rule-based)

Anchors produce high-precision IF-THEN rules:

- **Defaulting client** → rule anchored on `PAY_0 > 0`, `PAY_2 > 0`, `PAY_3 > 0`, `AGE > 34`, and several bill/payment thresholds. **Precision ≈ 0.81**, coverage ≈ 2.6%. (Consistent late payments across recent months anchor a "default" prediction.)
- **Non-defaulting client** → a single, broad rule: **`PAY_0 <= -1`** (i.e., paid off the most recent balance). **Precision ≈ 0.86**, coverage ≈ 27.9%. Paying on time alone reliably explains a "no default" prediction for a large slice of clients.

### Counterfactual explanations (DiCE)

The analysis reports DiCE counterfactual examples that change model predictions. These are model-behavior illustrations, not validated advice: categorical or personal-attribute changes may be infeasible or inappropriate, and a changed prediction does not establish a causal outcome.

### Fairness results

**Reported Fairlearn results** (zero indicates parity for the particular difference metric, not overall fairness):

| Sensitive attribute | Statistical Parity Diff. | Equal Opportunity Diff. | Reported comparison |
|---|---|---|---|
| **Gender** | 0.007 | 0.006 | Smallest reported gap |
| **Marital status** | 0.032 | 0.044 | Intermediate reported gap |
| **Age group** | 0.074 | 0.069 | Largest reported gap |

The age-group gap is driven by the **"Older" (>50) group**, which had a noticeably higher selection rate (0.19 vs. ~0.12) and false-positive rate (0.12 vs. ~0.06) than younger groups.

**Reported AIF360 results** (a disparate-impact ratio of 1 indicates parity for the specified selection-rate comparison):

| Comparison | Disparate Impact | Statistical Parity Diff. |
|---|---|---|
| **Gender** (Male vs. Female) | 1.014 | 0.012 |
| **Married vs. Single** | 1.011 | 0.010 |
| **Age: Middle-aged vs. Young** | 0.974 | −0.023 |
| **Age: Middle-aged vs. Older** | **0.931** | **−0.061** |

The written analysis reports larger gaps across age groups than across gender or marital status. These descriptive comparisons do not certify fairness. Review the source notebook, group definitions, favorable-outcome encoding, sample sizes, and uncertainty before interpreting these calculations.

### Summary of findings

1. Repayment and financial-history variables rank highly in the reported feature importance; this does not rule out demographic proxy effects.
2. **Recent repayment status (`PAY_0`)** is the dominant driver across every explanation method (importance, SHAP, Anchors).
3. Both models under-detect the minority "default" class (recall ≈ 0.48), the most consequential error type in lending.
4. The reported age-group disparities warrant further investigation; small gaps on other metrics are not proof of fairness.

## Run and inspect the analysis

Open [explainable_fair_credit.ipynb](explainable_fair_credit.ipynb), uploaded from `AI_Ethics_Final (1)(1).ipynb` with its original code and saved outputs preserved.

The notebook targets Google Colab and reads `/content/UCI_Credit_Card.csv` in several sections. Obtain the CSV from the dataset link above and upload it to Colab under that exact filename. For local Jupyter, replace the Colab-specific import and dataset paths. The CSV was not attached and is not included in this repository; saved preview rows cannot reconstruct the full dataset.

The notebook uses pandas, NumPy, matplotlib, seaborn, scikit-learn, statsmodels, imbalanced-learn, InterpretML, SHAP, LIME, Alibi, DiCE, Fairlearn, and AIF360. Versions are not pinned. Review the installation cells before running; the first contains `!pip install - q interpret alibi dice-ml`, whose option should be `-q`. A full rerun has not been performed.

### Interpretation and reproducibility notes

- The Fairlearn results labeled “Equal Opportunity Difference” call `equalized_odds_difference`; the earlier results table retains the original label and reported values. These metrics differ.
- The saved waterfall plots use `shap_values[instance_index, 0, class_index]` as the base value instead of the explainer's expected value. Treat their endpoints as unvalidated; correcting and regenerating them requires rerunning the model with the dataset.
- Different notebook sections retrain models with different splits and preprocessing. Compare metrics within their corresponding sections; Fairlearn treats class 1 (default) as positive, while AIF360 sets class 0 (no default) as favorable.
- The three plots in `images/` are exact copies of the existing `Result_Image/` files and match notebook cells 23, 31, and 32. The original directory is retained.

## Repository structure

```text
README.md
explainable_fair_credit.ipynb  Original notebook with saved outputs
images/                      SHAP summary and two waterfall plots
Result_Image/                Preserved original plots
LICENSE
```

`UCI_Credit_Card.csv` must be supplied separately.

## 📚 Key Libraries & References

- [InterpretML](https://interpret.ml/) — Explainable Boosting Machines
- [SHAP](https://shap.readthedocs.io/) — SHapley Additive exPlanations
- [LIME](https://github.com/marcotcr/lime) — Local Interpretable Model-agnostic Explanations
- [Alibi](https://docs.seldon.io/projects/alibi/) — Anchor explanations
- [DiCE](https://interpret.ml/DiCE/) — Diverse Counterfactual Explanations
- [Fairlearn](https://fairlearn.org/) — Fairness assessment toolkit
- [AIF360](https://aif360.res.ibm.com/) — AI Fairness 360 (IBM)

---

*This project was completed as the final project for INSY 5344 (AI Ethics). It demonstrates how interpretability and fairness tooling can be combined to audit a real-world credit risk model.*
