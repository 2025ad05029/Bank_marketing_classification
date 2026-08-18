"""
train.py
--------
Trains 5 classification models (Logistic Regression, Decision Tree, kNN,
Gaussian Naive Bayes, Random Forest) on the UCI Bank Marketing dataset
(bank-additional-full.csv). Saves each fitted pipeline (preprocessing +
model) to the model/ folder as a .pkl file, saves a held-out
test_data.csv (with the true label column so the Streamlit app can score
it), and prints/saves a metrics comparison table.

Dataset source: UCI Machine Learning Repository - Bank Marketing
(Moro, Cortez & Rita, 2014).
Target column: y (yes/no -> did the client subscribe to a term deposit)

IMPORTANT NOTE ON 'duration':
The UCI documentation flags 'duration' (last contact duration in seconds)
as a leakage feature: it is only known AFTER a call ends, and if
duration=0 the outcome is always 'no'. Including it would give an
unrealistically inflated, non-deployable model (it can't be known before
the call is placed). It is therefore DROPPED here to keep the model
usable for its actual real-world purpose: deciding whom to call.
"""

import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
df = pd.read_csv("bank_marketing.csv", sep=";")

# Drop 'duration' - known post-outcome leakage feature (see module docstring)
df = df.drop(columns=["duration"])

TARGET = "y"
df[TARGET] = df[TARGET].map({"yes": 1, "no": 0})

X = df.drop(columns=[TARGET])
y = df[TARGET]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("Numeric features    :", numeric_features)
print("Categorical features:", categorical_features)

# ---------------------------------------------------------------------
# 2. Train / test split (stratified, since classes are imbalanced ~89/11)
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Save the test split (features + true label) -> this is the file you will
# upload to the Streamlit app / submit as test_data.csv
test_export = X_test.copy()
test_export[TARGET] = y_test.values
test_export.to_csv("test_data.csv", index=False)
print(f"Saved test_data.csv with {len(test_export)} rows")

# ---------------------------------------------------------------------
# 3. Preprocessing: scale numeric, one-hot encode categorical
# ---------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ---------------------------------------------------------------------
# 4. Define the 5 required models
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=10),
    "kNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=150, max_depth=12, random_state=RANDOM_STATE
    ),
}

results = []
fitted_pipelines = {}

for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    results.append(metrics)
    fitted_pipelines[name] = pipe

    # save each fitted pipeline (preprocessing + model bundled together)
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    with open(f"model/{safe_name}.pkl", "wb") as f:
        pickle.dump(pipe, f)

    print(f"\n{name}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k}: {v:.4f}")

# ---------------------------------------------------------------------
# 5. Save comparison table (used in README + shown in Streamlit app)
# ---------------------------------------------------------------------
results_df = pd.DataFrame(results).set_index("Model").round(4)
results_df.to_csv("model/metrics_comparison.csv")
print("\n=== Comparison Table ===")
print(results_df)

# Also store feature lists so the Streamlit app knows the schema
schema = {
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "target": TARGET,
}
with open("model/schema.json", "w") as f:
    json.dump(schema, f, indent=2)

print("\nAll models trained and saved to model/ folder.")
