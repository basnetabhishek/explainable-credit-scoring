# Explainable AI & Fairness Analysis — Credit Card Default Prediction

**AI Ethics (INSY 5344) — Final Project**

This project applies Explainable AI (XAI) and fairness auditing techniques to a credit card default prediction task. Using the UCI Credit Card Default dataset, it trains machine learning models to predict whether a client will default on their next payment, then interrogates those models with a suite of interpretability tools and evaluates them for bias across gender, age, and marital status.

## 📊 Dataset

**[Default of Credit Card Clients (UCI / Taiwan)](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)** — 30,000 clients with demographic attributes (sex, age, education, marriage), credit limits, payment history, bill statements, and payment amounts.

- **Target:** `default.payment.next.month` (1 = default, 0 = no default)
- Class imbalance in the training set is handled with **SMOTE** oversampling.

> ⚠️ The dataset CSV (`UCI_Credit_Card.csv`) is not included in this repo. Download it from Kaggle or the UCI Machine Learning Repository and place it in your working directory (or upload it to `/content/` if running in Google Colab).

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

The model is audited for disparate treatment across three sensitive attributes using two industry-standard toolkits:

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

Reassuringly, **`SEX` is the least important feature**, and the model leans most heavily on actual payment behavior (`PAY_0`) — a strong signal that the model is learning financial risk rather than demographics.

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

DiCE generated realistic "what-if" scenarios showing the minimal changes needed to flip a prediction (e.g., adjusting repayment-status and marriage-category features to turn a predicted defaulter into a non-defaulter), giving clients concrete, if simplified, paths to a different outcome.

### Fairness results

**Fairlearn** (lower absolute values = more fair; 0 is perfect parity):

| Sensitive attribute | Statistical Parity Diff. | Equal Opportunity Diff. | Verdict |
|---|---|---|---|
| **Gender** | 0.007 | 0.006 | ✅ Essentially fair |
| **Marital status** | 0.032 | 0.044 | ⚠️ Mild disparity |
| **Age group** | 0.074 | 0.069 | ⚠️ Largest disparity |

The age-group gap is driven by the **"Older" (>50) group**, which had a noticeably higher selection rate (0.19 vs. ~0.12) and false-positive rate (0.12 vs. ~0.06) than younger groups.

**AIF360** (Disparate Impact closest to 1.0 = most fair):

| Comparison | Disparate Impact | Statistical Parity Diff. |
|---|---|---|
| **Gender** (Male vs. Female) | 1.014 | 0.012 |
| **Married vs. Single** | 1.011 | 0.010 |
| **Age: Middle-aged vs. Young** | 0.974 | −0.023 |
| **Age: Middle-aged vs. Older** | **0.931** | **−0.061** |

Both toolkits agree: the model is **very fair on gender and marital status**, but shows the **most meaningful disparity across age groups**, specifically disadvantaging older clients. This is the key fairness finding and the natural focus for any bias-mitigation follow-up.

### Summary of findings

1. The models rely on **financial behavior, not demographics** — `SEX` was the single least important feature.
2. **Recent repayment status (`PAY_0`)** is the dominant driver across every explanation method (importance, SHAP, Anchors).
3. Both models under-detect the minority "default" class (recall ≈ 0.48), the most consequential error type in lending.
4. Fairness is strong for **gender and marital status** but the model is **less fair across age groups**, particularly for **older clients** — the clearest area for improvement.

> 📌 The exact numbers above come from one run of the notebook. Re-running may shift values slightly depending on library versions and randomness, though `random_state=42` is set throughout for reproducibility.

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
pip install interpret shap lime alibi dice-ml
pip install imbalanced-learn fairlearn aif360 'aif360[all]'
```

## 🚀 How to Run

**Option 1 — Google Colab (recommended):**
1. Open `explainable_fair_credit.ipynb` in Colab
2. Upload `UCI_Credit_Card.csv` to the Colab session (`/content/`)
3. Run all cells — install commands are included in the notebook

**Option 2 — Local Jupyter:**
1. Install the requirements above
2. Update the CSV path in the data-loading cells (from `/content/UCI_Credit_Card.csv` to your local path)
3. Launch Jupyter and run the notebook top to bottom

## 📁 Repository Structure

```
├── explainable_fair_credit.ipynb   # Main notebook: modeling, XAI, and fairness analysis
├── README.md                       # This file
├── images/                         # Result plots referenced in this README
│   ├── shap_summary.png
│   ├── shap_waterfall_instance0.png
│   └── shap_waterfall_instance1.png
└── UCI_Credit_Card.csv             # Dataset (download separately — see above)
```

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
