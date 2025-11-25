import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

OUR_OPERATOR_NAME = "OurTel"
FULL_DATA_FILE = "telecom_churn_dataset_current_operator.csv"
OURTEL_ENRICHED_FILE = "telecom_churn_with_scores_ourtel.csv"
BEST_MODEL_FILE = "best_churn_model.pkl"


@st.cache_data
def load_full_data():
    if not os.path.exists(FULL_DATA_FILE):
        st.error(f"{FULL_DATA_FILE} not found. Run python_project.py once (locally or in backend) to generate it.")
        st.stop()
    return pd.read_csv(FULL_DATA_FILE)


@st.cache_data
def load_ourtel_data():
    if not os.path.exists(OURTEL_ENRICHED_FILE):
        st.error(f"{OURTEL_ENRICHED_FILE} not found. Run python_project.py once to generate scores.")
        st.stop()
    return pd.read_csv(OURTEL_ENRICHED_FILE)


@st.cache_resource
def load_model():
    if not os.path.exists(BEST_MODEL_FILE):
        st.error(f"{BEST_MODEL_FILE} not found. Run python_project.py once to train the model.")
        st.stop()
    return joblib.load(BEST_MODEL_FILE)


def main():
    st.set_page_config(page_title="OurTel Churn Dashboard", layout="wide")

    st.title("📊 OurTel Telecom Churn Dashboard")
    st.write("If you see this text, `app.py` is running correctly ✅")

    full_df = load_full_data()
    ourtel_df = load_ourtel_data()
    _ = load_model()

    # Ensure only OurTel customers in enriched file
    ourtel_df = ourtel_df[ourtel_df["current_operator"] == OUR_OPERATOR_NAME].copy()

    # ----- KPIs -----
    total_ourtel = len(ourtel_df)
    ourtel_churn_rate = ourtel_df["churn"].mean() if total_ourtel else 0.0
    grey_share = (ourtel_df["risk_segment"] == "Grey area").mean() if total_ourtel else 0.0
    high_share = (ourtel_df["risk_segment"] == "High risk").mean() if total_ourtel else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("OurTel Customers", f"{total_ourtel:,}")
    c2.metric("OurTel Churn Rate", f"{ourtel_churn_rate:.1%}")
    c3.metric("Grey-area Customers", f"{grey_share:.1%}")
    c4.metric("High-risk Customers", f"{high_share:.1%}")

    st.markdown("---")

    # ----- Sidebar filters -----
    st.sidebar.header("Filters (OurTel only)")

    regions = sorted(ourtel_df["region"].dropna().unique().tolist())
    risk_segments = sorted(ourtel_df["risk_segment"].dropna().unique().tolist())
    plan_types = sorted(ourtel_df["plan_type"].dropna().unique().tolist())

    selected_regions = st.sidebar.multiselect("Region", regions, default=regions)
    selected_risk = st.sidebar.multiselect("Risk segment", risk_segments, default=risk_segments)
    selected_plan = st.sidebar.multiselect("Plan type", plan_types, default=plan_types)

    ourtel_filtered = ourtel_df[
        ourtel_df["region"].isin(selected_regions)
        & ourtel_df["risk_segment"].isin(selected_risk)
        & ourtel_df["plan_type"].isin(selected_plan)
    ]

    st.subheader("Filtered OurTel Segment Overview")
    st.write(f"Customers in filter: **{len(ourtel_filtered):,}**")
    if len(ourtel_filtered):
        st.write(f"Churn rate in filter: **{ourtel_filtered['churn'].mean():.1%}**")
    else:
        st.write("Churn rate in filter: N/A")

    # Tabs
    tab_overview, tab_risk = st.tabs(["🌍 Market Overview", "📉 OurTel Risk & Actions"])

    # ---- TAB 1: Market overview (all operators) ----
    with tab_overview:
        st.markdown("### Churn by Operator and Region (all customers)")

        if {"region", "current_operator", "churn"}.issubset(full_df.columns):
            op_region = (
                full_df.groupby(["region", "current_operator"])["churn"]
                .mean()
                .reset_index()
            )
            pivot = op_region.pivot(index="region", columns="current_operator", values="churn")

            st.write("Churn rate table:")
            if not pivot.empty:
                st.dataframe(pivot.style.format("{:.1%}"))
            else:
                st.dataframe(pivot)

            op_churn = (
                full_df.groupby("current_operator")["churn"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            fig, ax = plt.subplots()
            ax.bar(op_churn["current_operator"], op_churn["churn"])
            ax.set_ylabel("Churn rate")
            ax.set_title("Churn Rate by Operator")
            ax.set_xticklabels(op_churn["current_operator"], rotation=45)
            st.pyplot(fig)
        else:
            st.info("Columns 'region', 'current_operator', 'churn' not all present in full dataset.")

    # ---- TAB 2: OurTel risk & actions ----
    with tab_risk:
        st.markdown("### OurTel Risk Segmentation")

        if len(ourtel_filtered) == 0:
            st.warning("No OurTel customers for current filter.")
        else:
            risk_counts = ourtel_filtered["risk_segment"].value_counts().reindex(
                ["Low risk", "Grey area", "High risk"]
            )
            fig_r, ax_r = plt.subplots()
            ax_r.bar(risk_counts.index, risk_counts.values)
            ax_r.set_ylabel("Number of customers")
            ax_r.set_title("OurTel Customers by Risk Segment")
            st.pyplot(fig_r)

            if "churn_probability" in ourtel_filtered.columns:
                fig_p, ax_p = plt.subplots()
                ax_p.hist(ourtel_filtered["churn_probability"], bins=30, edgecolor="black")
                ax_p.set_xlabel("Churn probability")
                ax_p.set_ylabel("Frequency")
                ax_p.set_title("Churn Probability Distribution (Filtered OurTel)")
                st.pyplot(fig_p)

            st.markdown("### Top Grey & High-risk OurTel Customers")
            risky = (
                ourtel_filtered[
                    ourtel_filtered["risk_segment"].isin(["Grey area", "High risk"])
                ]
                .sort_values("churn_probability", ascending=False)
                .head(100)
            )

            cols_to_show = [
                "customer_id",
                "region",
                "plan_type",
                "plan_category",
                "monthly_charge",
                "call_drops",
                "network_issues",
                "dissatisfaction_score",
                "churn_probability",
                "risk_segment",
                "retention_actions",
            ]
            cols_to_show = [c for c in cols_to_show if c in risky.columns]
            st.dataframe(risky[cols_to_show])


if __name__ == "__main__":
    main()
