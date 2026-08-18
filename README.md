# Bank Marketing — Term Deposit Subscription Classification

M.Tech (AIML/DSE) — Machine Learning — Assignment 2

## a. Problem Statement

Banks running telemarketing campaigns want to know, before making a call,
which customers are most likely to subscribe to a term deposit. Calling
every customer is expensive and time-consuming; being able to rank
customers by predicted likelihood of subscribing lets the bank prioritize
its calling list and improve campaign efficiency.

This project frames it as a **binary classification problem**: given a
customer's demographic profile, financial attributes, and details of the
current/previous marketing campaign contacts, predict whether they will
subscribe to a term deposit (`y = yes`) or not (`y = no`).

## b. Dataset Description

**Source:** UCI Machine Learning Repository — *Bank Marketing* dataset
(`bank-additional-full.csv`), based on a Portuguese banking institution's
phone-call marketing campaigns (Moro, Cortez & Rita, 2014).

- **Instances:** 41,188 customer contacts
- **Features used:** 19 (20 raw columns minus `duration`, see note below —
  still well above the minimum required 12)
- **Target:** `y` (yes / no — subscribed to a term deposit)
- **Class balance:** 36,548 "no" vs. 4,640 "yes" — a strongly imbalanced
  ~89% / 11% split, typical of real-world marketing conversion data.

**Feature groups:**
| Type | Features |
|---|---|
| Demographics | `age`, `job`, `marital`, `education` |
| Financial | `default`, `housing`, `loan` |
| Current campaign contact | `contact`, `month`, `day_of_week`, `campaign` |
| Previous campaign history | `pdays`, `previous`, `poutcome` |
| Macroeconomic context | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` |

**Important design decision — dropping `duration`:** the UCI
documentation explicitly flags `duration` (the last call's length in
seconds) as a leakage feature: it is only known *after* the call ends,
and a `duration` of 0 always means `y = no`. Keeping it in would make the
model look artificially strong while being useless for its real purpose
— deciding *whom to call in the first place*. It was therefore dropped
before training, and this project reports honest, deployable metrics
rather than inflated ones.

No missing values were present in the raw data. Numeric features were
standardized (`StandardScaler`) and categorical features were one-hot
encoded inside a single `sklearn` `Pipeline`, so identical preprocessing
is applied at training and inference time in the Streamlit app.

A stratified 80/20 train/test split (`random_state=42`) preserves the
class ratio in both splits. The 20% hold-out split (8,238 rows) is saved
as `test_data.csv` and is what gets uploaded to the Streamlit app for
scoring.

## c. GitHub Repository Link

`<PASTE YOUR GITHUB REPO URL HERE AFTER YOU PUSH>`

## d. Models Used

All 5 models were trained on identical preprocessed data (`train.py`)
and evaluated on the same hold-out test split.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9009 | 0.8008 | 0.6905 | 0.2188 | 0.3322 | 0.3516 |
| Decision Tree | 0.9013 | 0.7581 | 0.6353 | 0.2909 | 0.3991 | 0.3856 |
| kNN | 0.8996 | 0.7611 | 0.6145 | 0.2920 | 0.3959 | 0.3775 |
| Naive Bayes | 0.8049 | 0.7755 | 0.3172 | **0.6347** | 0.4230 | 0.3490 |
| Random Forest (Ensemble) | **0.9020** | **0.8116** | **0.6817** | 0.2446 | 0.3600 | **0.3694** |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Solid precision but low recall (0.22) — the model is conservative, only flagging customers it's confident about, missing many real subscribers. |
| Decision Tree | Best F1 among the non-Naive-Bayes models, but the lowest AUC, since a single tree overfits and ranks borderline cases poorly. |
| kNN | Very similar profile to Decision Tree; distance-based similarity is diluted by the many one-hot encoded categorical features (job, month, etc.), which limits its ranking ability (AUC). |
| Naive Bayes | By far the highest recall (0.63) — its independence assumption makes it much more willing to flag "yes", catching more real subscribers at the cost of many false positives (lowest precision, lowest accuracy). |
| Random Forest (Ensemble) | **Best overall** on Accuracy, AUC, Precision, and MCC — averaging many trees generalizes better than a single tree or a linear model on this feature mix, though its recall (0.24) is still limited by the severe class imbalance. |
| **Overall Winner** | **Random Forest (Ensemble)** — best AUC and MCC, meaning it ranks customers by subscription likelihood most reliably, which is exactly what the bank needs to prioritize its call list. All models show low recall due to the ~89/11 class imbalance and the removal of the leaky `duration` feature — a class-imbalance technique (e.g. SMOTE or class-weighting) would be a natural next step to explore. |

## Repository Structure

```
project-folder/
│-- app.py                       # Streamlit app
│-- train.py                     # Trains all 5 models, saves pipelines + metrics
│-- requirements.txt
│-- runtime.txt                  # Pins Python 3.11 for consistent deployment
│-- README.md
│-- test_data.csv                # Held-out test split (upload this in the app)
│-- bank_marketing.csv           # Full raw dataset (for reference)
│-- model/
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest_ensemble.pkl
│   │-- metrics_comparison.csv
│   └-- schema.json
```

## Streamlit App Features

- 📁 CSV upload (upload `test_data.csv`)
- ⚙️ Model selection dropdown (choose among the 5 trained models)
- 📈 Live evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- 🔢 Confusion matrix + classification report
- 📊 Pre-computed comparison table across all models
- ⬇️ Download predictions as CSV

## How to Run Locally

```bash
pip install -r requirements.txt
python train.py          # retrains models (optional — .pkl files already included)
streamlit run app.py
```

## Live App

`<PASTE YOUR STREAMLIT COMMUNITY CLOUD LINK HERE>`
