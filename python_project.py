# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Try to import matplotlib; if not available (e.g. on Streamlit Cloud), just skip plots
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ModuleNotFoundError:
    HAS_MPL = False
    print("matplotlib not installed – EDA plots will be skipped.")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import joblib

# -----------------------------
# Global constants
# -----------------------------
DATA_FILE = "telecom_churn_dataset_current_operator.csv"
OUR_OPERATOR_NAME = "OurTel"
ENRICHED_OURTEL_FILE = "telecom_churn_with_scores_ourtel.csv"
BEST_MODEL_FILE = "best_churn_model.pkl"


# -----------------------------
# 1️⃣ EDA
# -----------------------------
def plot_histogram(df, column, bins=30):
    """Plot a histogram, if matplotlib is available."""
    if not HAS_MPL:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(df[column].dropna(), bins=bins, edgecolor="black")
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_correlation_heatmap(df):
    """Plot correlation heatmap, if matplotlib is available."""
    if not HAS_MPL:
        return

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("No numeric columns for correlation heatmap.")
        return

    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr_matrix, interpolation="nearest", aspect="auto")
    plt.title("Correlation Heatmap")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    ticks = np.arange(len(corr_matrix.columns))
    plt.xticks(ticks, corr_matrix.columns, rotation=90)
    plt.yticks(ticks, corr_matrix.columns)

    plt.tight_layout()
    plt.show()
    plt.close()


def run_eda(df: pd.DataFrame) -> None:
    """Print basic EDA summary and (optionally) plots."""
    print("===== BASIC INFO =====")
    print("Shape:", df.shape)
    print("\nHead:")
    print(df.head())
    print("\nInfo:")
    df.info()
    print("\nDescribe:")
    print(df.describe())

    if "churn" in df.columns:
        print("\nOverall churn rate:", df["churn"].mean())
    else:
        print("'churn' column not found.")
        return

    if "region" in df.columns:
        print("\nChurn by region:")
        print(df.groupby("region")["churn"].mean())

    if "current_operator" in df.columns:
        print("\nChurn by current operator:")
        print(df.groupby("current_operator")["churn"].mean())

    if "plan_type" in df.columns and "plan_category" in df.columns:
        print("\nChurn by plan type & category:")
        print(df.groupby(["plan_type", "plan_category"])["churn"].mean())

    if "region" in df.columns and "current_operator" in df.columns:
        print("\nChurn by region & operator:")
        table = (
            df.groupby(["region", "current_operator"])["churn"]
            .mean()
            .unstack()
        )
        print(table)

    # Plots only if matplotlib is available
    if HAS_MPL:
        hist_cols = [
            "tenure_months",
            "monthly_charge",
            "call_drops",
            "network_issues",
            "dissatisfaction_score",
        ]
        for col in hist_cols:
            if col in df.columns:
                plot_histogram(df, col)

        plot_correlation_heatmap(df)
    else:
        print("\nSkipping plots because matplotlib is not installed.")


# -----------------------------
# 2️⃣ MODEL TRAINING
# -----------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def train_and_compare_models(df_ourtel: pd.DataFrame):
    """Train Logistic Regression & Random Forest on OurTel customers and pick the best model."""
    df_ourtel = df_ourtel[df_ourtel["current_operator"] == OUR_OPERATOR_NAME].copy()

    X = df_ourtel.drop(columns=["customer_id", "churn", "current_operator"])
    y = df_ourtel["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        )
    }

    results = []
    fitted_models = {}

    print("\n===== MODEL TRAINING =====")

    for name, clf in models.items():
        print(f"\nTraining {name}...")
        pipe = Pipeline([("preprocessor", preprocessor), ("model", clf)])
        pipe.fit(X_train, y_train)
        fitted_models[name] = pipe

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc:.4f}")
        print("Confusion Matrix:\n", cm)

        results.append({
            "model": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc
        })

    metrics_df = pd.DataFrame(results).set_index("model")
    print("\n===== MODEL COMPARISON =====")
    print(metrics_df)

    best_name = metrics_df["roc_auc"].idxmax()
    best_model = fitted_models[best_name]
    print("\nBest model:", best_name)

    joblib.dump(best_model, BEST_MODEL_FILE)
    print("Saved best model to:", BEST_MODEL_FILE)

    return best_model, metrics_df


# -----------------------------
# 3️⃣ SCORING + RISK
# -----------------------------
def assign_risk_segment(prob: float) -> str:
    if prob < 0.30:
        return "Low risk"
    elif prob <= 0.70:
        return "Grey area"
    return "High risk"


def score_and_segment(df_ourtel: pd.DataFrame, best_model: Pipeline) -> pd.DataFrame:
    """Score OurTel customers, assign churn_probability + risk_segment."""
    df_ourtel = df_ourtel[df_ourtel["current_operator"] == OUR_OPERATOR_NAME].copy()

    X = df_ourtel.drop(columns=["customer_id", "churn", "current_operator"])
    probs = best_model.predict_proba(X)[:, 1]

    df_ourtel["churn_probability"] = probs
    df_ourtel["risk_segment"] = df_ourtel["churn_probability"].apply(assign_risk_segment)

    print("\nRisk segment distribution:")
    print(df_ourtel["risk_segment"].value_counts())
    print("\nRisk segment proportions:")
    print(df_ourtel["risk_segment"].value_counts(normalize=True))
    print("\nActual churn by segment:")
    print(df_ourtel.groupby("risk_segment")["churn"].mean())

    return df_ourtel


# -----------------------------
# 4️⃣ RETENTION ACTIONS
# -----------------------------
def retention_action(row) -> str:
    actions = []

    if row.get("call_drops", 0) >= 3:
        actions.append("Optimize network; give free voice minutes.")
    if row.get("network_issues", 0) >= 5:
        actions.append("Provide free data voucher; prioritize network fix.")
    if row.get("dissatisfaction_score", 0) >= 7:
        actions.append("Proactive support call + goodwill credit.")
    if row.get("monthly_charge", 0) >= 600 and row.get("tenure_months", 0) >= 12:
        actions.append("Offer loyalty rewards and VIP care.")
    if row.get("risk_segment") == "High risk":
        actions.append("Urgent intervention by relationship manager.")
    elif row.get("risk_segment") == "Grey area":
        actions.append("Send targeted app/SMS offers.")

    if not actions:
        actions.append("Maintain engagement with periodic check-ins.")

    return " ".join(actions)


def add_retention_actions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["retention_actions"] = df.apply(retention_action, axis=1)
    df.to_csv(ENRICHED_OURTEL_FILE, index=False)
    print("\nSaved enriched dataset to:", ENRICHED_OURTEL_FILE)
    return df


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Loading dataset:", DATA_FILE)
    df = pd.read_csv(DATA_FILE)

    print("\nRunning EDA…")
    run_eda(df)

    df_ourtel = df[df["current_operator"] == OUR_OPERATOR_NAME].copy()

    print("\nTraining models…")
    best_model, _ = train_and_compare_models(df_ourtel)

    print("\nScoring customers…")
    scored = score_and_segment(df_ourtel, best_model)

    print("\nAdding retention recommendations…")
    add_retention_actions(scored)

    print("\nDONE!")
    print("Best model saved at:", BEST_MODEL_FILE)
    print("Enriched dataset saved at:", ENRICHED_OURTEL_FILE)


if __name__ == "__main__":
    main()
