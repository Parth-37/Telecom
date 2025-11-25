# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

# Try to import matplotlib; skip plots if not available
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
RANDOM_SEED = 42
N_CUSTOMERS = 50000


# -----------------------------
# 0️⃣ DATA GENERATION (if missing)
# -----------------------------
def generate_dataset() -> pd.DataFrame:
    """Generate a synthetic telecom churn dataset with current_operator + region."""
    np.random.seed(RANDOM_SEED)

    df = pd.DataFrame({
        "customer_id": np.arange(1, N_CUSTOMERS + 1),

        # Demographics & Region
        "age": np.random.randint(18, 70, N_CUSTOMERS),
        "region": np.random.choice(
            ["North", "South", "East", "West", "Central", "North-East", "Metro"],
            N_CUSTOMERS,
            p=[0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.10]
        ),

        # Tenure & Revenue
        "tenure_months": np.random.randint(1, 72, N_CUSTOMERS),
        "monthly_charge": np.round(np.random.uniform(100, 1200, N_CUSTOMERS), 2),

        # Usage & Quality
        "call_drops": np.random.poisson(2, N_CUSTOMERS),
        "network_issues": np.random.randint(0, 10, N_CUSTOMERS),
        "data_speed_rating": np.random.randint(1, 6, N_CUSTOMERS),

        # Support-Related
        "customer_service_calls": np.random.poisson(1, N_CUSTOMERS),
        "ticket_count_last_6_months": np.random.poisson(2, N_CUSTOMERS),
        "issue_resolution_time_avg": np.random.uniform(2, 72, N_CUSTOMERS),  # hours
        "customer_support_rating": np.random.randint(1, 6, N_CUSTOMERS),

        # Current Operator (who they are using now)
        "current_operator": np.random.choice(
            ["OurTel", "Airtel", "Jio", "Vodafone", "BSNL"],
            N_CUSTOMERS,
            p=[0.35, 0.20, 0.20, 0.15, 0.10],  # OurTel is "us"
        ),

        # Plan Attributes
        "plan_type": np.random.choice(["Prepaid", "Postpaid"], N_CUSTOMERS, p=[0.60, 0.40]),
        "plan_category": np.random.choice(["Unlimited", "Data-only", "Voice-heavy", "Combo"], N_CUSTOMERS),
        "addons_subscribed": np.random.choice(
            ["None", "Prime", "Hotstar", "Netflix"],
            N_CUSTOMERS,
            p=[0.50, 0.20, 0.20, 0.10]
        ),
        "validity_days": np.random.choice([28, 56, 84, 365], N_CUSTOMERS, p=[0.40, 0.30, 0.20, 0.10]),

        # Satisfaction
        "dissatisfaction_score": np.random.randint(1, 11, N_CUSTOMERS),
    })

    # Derived revenue
    df["total_charge"] = df["monthly_charge"] * df["tenure_months"]

    # Operator-specific baseline churn risk (so some operators churn more)
    operator_risk = {
        "OurTel": 0.08,
        "Airtel": 0.10,
        "Jio": 0.12,
        "Vodafone": 0.11,
        "BSNL": 0.15
    }
    df["operator_risk"] = df["current_operator"].map(operator_risk)

    # Churn probability model (synthetic but realistic)
    prob = (
        0.15 * (df["call_drops"] > 3).astype(int) +
        0.20 * (df["network_issues"] > 5).astype(int) +
        0.25 * (df["dissatisfaction_score"] > 7).astype(int) +
        0.10 * (df["ticket_count_last_6_months"] > 3).astype(int) +
        0.10 * (df["issue_resolution_time_avg"] > 36).astype(int) +
        df["operator_risk"]
    )

    noise = np.random.normal(0, 0.1, N_CUSTOMERS)
    df["churn"] = np.where(prob + noise > 0.6, 1, 0)

    df = df.drop(columns=["operator_risk"])

    return df


def load_or_generate_data() -> pd.DataFrame:
    """Load dataset from CSV if present, otherwise generate and save it."""
    if os.path.exists(DATA_FILE):
        print(f"Found existing dataset: {DATA_FILE}")
        return pd.read_csv(DATA_FILE)

    print(f"{DATA_FILE} not found. Generating synthetic dataset...")
    df = generate_dataset()
    df.to_csv(DATA_FILE, index=False)
    print(f"Synthetic dataset saved to {DATA_FILE}")
    return df


# -----------------------------
# 1️⃣ EDA
# -----------------------------
def plot_histogram(df, column, bins=30):
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
    df_ourtel = df_ourtel[df_ourtel["current_operator"] == OUR_OPERATOR_NAME].copy()

    X = df_ourtel.drop(columns=["customer_id", "churn", "current_operator"])
    y = df_ourtel["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    preprocessor = build_preprocessor(X)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1
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
    print("Loading or generating dataset...")
    df = load_or_generate_data()

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
