"""
Employee Attrition Risk — Streamlit application.

Loads the trained pipeline (best_model.pkl) and metadata (metadata.pkl)
produced by pipeline.ipynb and serves an HR-facing interface with
prediction, SHAP explanation, and model insight pages.

Run locally:    streamlit run app.py
Deployed at:    <https://YOUR-APP-URL.streamlit.app>
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Risk",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

FIG_DIR = Path("figures")


# ----------------------------------------------------------------------
# Cached resource loaders — load .pkl files from the same folder as app.py
# (following course convention: model file sits beside app.py)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_metadata():
    with open("metadata.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_metrics():
    with open("metrics.json", "r") as f:
        return json.load(f)


@st.cache_resource
def get_explainer(_pipeline):
    """Build a SHAP explainer for the classifier inside the pipeline."""
    clf = _pipeline.named_steps["clf"]
    try:
        return shap.TreeExplainer(clf), _pipeline.named_steps["pre"], "tree"
    except Exception:
        # Fallback for non-tree models (e.g. logistic regression, SVM)
        pre = _pipeline.named_steps["pre"]
        bg_df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv").drop(
            columns=[c for c in ["EmployeeNumber", "EmployeeCount",
                                 "StandardHours", "Over18", "Attrition"]
                     if c in pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv").columns],
            errors="ignore",
        ).sample(50, random_state=42)
        bg_trans = pre.transform(bg_df)
        return shap.KernelExplainer(clf.predict_proba, bg_trans), pre, "kernel"


def get_feature_names(pipeline, meta):
    pre = pipeline.named_steps["pre"]
    ohe = pre.named_transformers_["cat"]
    return meta["numerical_cols"] + list(
        ohe.get_feature_names_out(meta["categorical_cols"])
    )


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("👥 Employee Attrition Risk Predictor")
st.caption(
    "Predicts the probability that an employee will leave the company, with an "
    "explanation of the key factors driving each prediction. "
    "Built for the COM763 Portfolio Task."
)

try:
    model = load_model()
    meta = load_metadata()
    metrics = load_metrics()
except FileNotFoundError:
    st.error(
        "Model files not found. Make sure best_model.pkl and metadata.pkl "
        "are in the same folder as app.py. Run pipeline.ipynb first to "
        "produce them."
    )
    st.stop()

# Top-line model stats in the sidebar
with st.sidebar:
    st.header("Model")
    st.metric("Model", meta["model_name"])
    st.metric("ROC-AUC (test)", f"{metrics['ROC-AUC']:.3f}")
    st.metric("PR-AUC (test)",  f"{metrics['PR-AUC']:.3f}")
    st.metric("F1 (test)",      f"{metrics['F1']:.3f}")
    st.divider()
    st.caption(
        f"Optimal F1 threshold: **{meta['best_threshold']:.2f}**. "
        "Adjust below to trade recall for precision."
    )
    threshold = st.slider(
        "Decision threshold",
        min_value=0.05, max_value=0.95,
        value=float(meta["best_threshold"]), step=0.01,
    )

tab_predict, tab_explain, tab_insights, tab_about = st.tabs(
    ["🔮 Predict", "🔍 Explain", "📊 Model insights", "ℹ️ About"]
)


# ----------------------------------------------------------------------
# 1. PREDICT
# ----------------------------------------------------------------------
with tab_predict:
    st.subheader("Enter employee details")
    st.caption("Defaults are populated from the dataset median / first option.")

    num_ranges = meta["numerical_ranges"]
    cat_options = meta["categorical_options"]

    inputs = {}
    cols = st.columns(3)
    all_fields = meta["numerical_cols"] + meta["categorical_cols"]
    for i, field in enumerate(all_fields):
        col = cols[i % 3]
        with col:
            if field in num_ranges:
                lo, hi, med = num_ranges[field]
                if float(int(lo)) == lo and float(int(hi)) == hi:
                    inputs[field] = st.number_input(
                        field,
                        min_value=int(lo), max_value=int(hi),
                        value=int(med), step=1,
                    )
                else:
                    inputs[field] = st.number_input(
                        field,
                        min_value=float(lo), max_value=float(hi),
                        value=float(med), step=0.1,
                    )
            else:
                inputs[field] = st.selectbox(field, cat_options[field])

    st.divider()
    if st.button("Predict attrition risk", type="primary", use_container_width=True):
        x = pd.DataFrame([inputs])
        prob = model.predict_proba(x)[0, 1]
        pred = int(prob >= threshold)

        c1, c2, c3 = st.columns([1, 1, 1])
        c1.metric("Attrition probability", f"{prob:.1%}")
        c2.metric("Threshold", f"{threshold:.2f}")
        c3.metric(
            "Prediction",
            "AT RISK" if pred else "Likely to stay",
            delta="Intervene" if pred else "No action",
            delta_color="inverse" if pred else "normal",
        )

        st.progress(min(float(prob), 1.0))
        if pred:
            st.error(
                f"⚠️ This employee's predicted attrition risk ({prob:.1%}) is "
                f"above the threshold ({threshold:.2f}). HR should consider a "
                "retention conversation."
            )
        else:
            st.success(
                f"✅ Predicted risk ({prob:.1%}) is below the threshold "
                f"({threshold:.2f}). No immediate intervention indicated."
            )

        st.session_state["last_input"] = x
        st.session_state["last_prob"] = float(prob)
        st.info("Open the **Explain** tab to see why the model made this prediction.")


# ----------------------------------------------------------------------
# 2. EXPLAIN
# ----------------------------------------------------------------------
with tab_explain:
    st.subheader("Why did the model predict this?")
    if "last_input" not in st.session_state:
        st.info("Make a prediction on the **Predict** tab first.")
    else:
        x = st.session_state["last_input"]
        prob = st.session_state["last_prob"]
        st.write(f"Predicted attrition probability: **{prob:.1%}**")

        with st.spinner("Computing SHAP values..."):
            explainer, pre, kind = get_explainer(model)
            x_trans = pre.transform(x)
            feature_names = get_feature_names(model, meta)

            if kind == "tree":
                shap_values = explainer.shap_values(x_trans)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                shap_values = explainer.shap_values(x_trans, nsamples=100)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

        contrib = pd.DataFrame({
            "feature": feature_names,
            "value": x_trans[0],
            "shap": shap_values[0] if shap_values.ndim > 1 else shap_values,
        })
        contrib["abs_shap"] = contrib["shap"].abs()
        top = contrib.nlargest(10, "abs_shap").sort_values("shap")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = ["#ef4444" if v > 0 else "#3b82f6" for v in top["shap"]]
        ax.barh(top["feature"], top["shap"], color=colors)
        ax.set_xlabel("SHAP value (→ pushes toward 'leave')")
        ax.set_title("Top 10 features driving this prediction")
        ax.axvline(0, color="black", linewidth=0.5)
        st.pyplot(fig, clear_figure=True)

        st.caption(
            "🔴 Red bars push the prediction toward **leave**, "
            "🔵 blue bars push it toward **stay**. Bar length = strength of effect."
        )


# ----------------------------------------------------------------------
# 3. MODEL INSIGHTS
# ----------------------------------------------------------------------
with tab_insights:
    st.subheader("Model performance on the held-out test set")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{metrics['Accuracy']:.3f}")
    c2.metric("Precision", f"{metrics['Precision']:.3f}")
    c3.metric("Recall",    f"{metrics['Recall']:.3f}")
    c4.metric("F1",        f"{metrics['F1']:.3f}")

    st.divider()
    st.markdown("#### Evaluation visuals")
    figs = [
        ("ROC and Precision-Recall curves",                    "07_roc_pr.png"),
        ("Confusion matrix at default threshold",              "06_confusion_matrix.png"),
        ("Threshold tuning (precision / recall / F1)",         "08_threshold_tuning.png"),
        ("Cross-validated model comparison",                   "05_cv_boxplot.png"),
        ("Global SHAP feature importance",                     "09_shap_importance.png"),
        ("SHAP beeswarm — per-feature impact direction",       "10_shap_beeswarm.png"),
    ]
    for title, fname in figs:
        path = FIG_DIR / fname
        if path.exists():
            st.markdown(f"**{title}**")
            st.image(str(path), use_container_width=True)


# ----------------------------------------------------------------------
# 4. ABOUT
# ----------------------------------------------------------------------
with tab_about:
    st.subheader("About this system")
    st.markdown(f"""
**Problem.** Employee attrition is costly — replacing a single employee can cost
50–200% of their annual salary in recruitment, onboarding and lost productivity.
HR teams need to identify at-risk employees *early* so they can intervene.

**Task.** Binary classification: predict whether an employee will leave
(`Attrition = Yes`) or stay. Framed as a *ranked-risk* problem rather than a
hard yes/no — HR uses the probability to triage interventions, and the
threshold can be tuned to match their available bandwidth.

**Data.** IBM HR Analytics Employee Attrition & Performance dataset:
1,470 employees, 35 features. Class imbalance ~16% positive class. Synthetic,
created by IBM data scientists — used widely as a benchmark.

**Model.** **{meta['model_name']}** chosen via 5-fold cross-validated ROC-AUC
on a stratified 80/20 train/test split. Class imbalance handled with
class-weighting (compared against SMOTE — class-weighting won on simplicity
and equal AUC). Hyperparameters tuned with `GridSearchCV`. Threshold tuned to
maximise F1 on the test set ({meta['best_threshold']:.2f}).

**Held-out test performance.**
ROC-AUC = **{metrics['ROC-AUC']:.3f}**, PR-AUC = **{metrics['PR-AUC']:.3f}**,
F1 = **{metrics['F1']:.3f}**, Recall = **{metrics['Recall']:.3f}**.

**Limitations & ethical considerations.**
- The dataset is synthetic, so generalisation to a specific organisation's
  workforce is not guaranteed. The system should be re-trained on real,
  representative data before any operational use.
- Predictions could entrench existing biases if used to make consequential
  decisions about individuals. The intended use is *decision support* — not
  automated action — and predictions should be combined with manager judgement.
- Sensitive attributes (gender, age, marital status) are present in the
  features. A production deployment should include a fairness audit
  (e.g. equalised odds across protected groups).

**Repo & report.** See accompanying GitHub repository and technical report.
""")
