import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
from sklearn.preprocessing import LabelEncoder
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic — XGBoost vs LightGBM Benchmark",
    page_icon="🚢",
    layout="wide",
)

st.title("🚢 Titanic Survival — XGBoost vs LightGBM Benchmark")
st.markdown(
    "Compare **XGBoost** and **LightGBM** on the classic Titanic dataset. "
    "Tune hyper-parameters in the sidebar, then click **Run Benchmark**."
)

# ──────────────────────────────────────────────
# Sidebar — hyper-parameters
# ──────────────────────────────────────────────
st.sidebar.header("⚙️ Model Hyper-parameters")
n_estimators  = st.sidebar.slider("Number of trees", 50, 500, 100, step=50)
max_depth     = st.sidebar.slider("Max depth", 2, 12, 6)
learning_rate = st.sidebar.slider("Learning rate", 0.01, 0.30, 0.10, step=0.01)
test_size     = st.sidebar.slider("Test size %", 10, 40, 20, step=5)
random_state  = st.sidebar.number_input("Random state", value=42, step=1)

# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
st.sidebar.header("📂 Data")
use_upload = st.sidebar.checkbox("Upload my own CSV", value=False)

if use_upload:
    uploaded = st.sidebar.file_uploader("Upload Titanic train CSV", type="csv")
    if uploaded is None:
        st.info("⬆️ Upload a CSV file to get started.")
        st.stop()
    df = pd.read_csv(uploaded)
else:
    # Use the bundled Titanic_train.csv
    import os
    csv_path = os.path.join(os.path.dirname(__file__), "Titanic_train.csv")
    if not os.path.exists(csv_path):
        st.error(f"Could not find `Titanic_train.csv` in {os.path.dirname(__file__)}")
        st.stop()
    df = pd.read_csv(csv_path)

# ──────────────────────────────────────────────
# Data preview
# ──────────────────────────────────────────────
with st.expander("📋 Raw Data Preview", expanded=False):
    st.dataframe(df.head(10), width="stretch")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    with col_b:
        st.markdown(f"**Survived distribution:** {dict(df['Survived'].value_counts())}")

# ──────────────────────────────────────────────
# Pre-processing
# ──────────────────────────────────────────────
def preprocess(dataframe):
    """Clean & encode the Titanic dataset."""
    data = dataframe.copy()

    # Drop columns that are not useful for modelling
    drop_cols = [c for c in ["PassengerId", "Name", "Ticket", "Cabin"] if c in data.columns]
    data.drop(columns=drop_cols, inplace=True)

    # Fill missing values
    data["Age"] = data["Age"].fillna(data["Age"].median())
    if "Fare" in data.columns:
        data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    if "Embarked" in data.columns:
        data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

    # Encode categorical columns
    le_sex = LabelEncoder()
    data["Sex"] = le_sex.fit_transform(data["Sex"])           # female=0, male=1

    if "Embarked" in data.columns:
        le_emb = LabelEncoder()
        data["Embarked"] = le_emb.fit_transform(data["Embarked"])  # C=0, Q=1, S=2

    return data

df_clean = preprocess(df)

with st.expander("🔧 Processed Data Preview", expanded=False):
    st.dataframe(df_clean.head(10), width="stretch")
    st.markdown(f"**Remaining nulls:** {df_clean.isnull().sum().sum()}")

# ──────────────────────────────────────────────
# Split
# ──────────────────────────────────────────────
TARGET = "Survived"
X = df_clean.drop(columns=[TARGET])
y = df_clean[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100, random_state=random_state, stratify=y
)

st.markdown(f"**Train:** {X_train.shape[0]} samples  •  **Test:** {X_test.shape[0]} samples")

# ──────────────────────────────────────────────
# Benchmark
# ──────────────────────────────────────────────
if st.button("🚀 Run Benchmark", type="primary", use_container_width=True):

    results = {}
    models  = {}

    # ── XGBoost ──────────────────────────────
    with st.spinner("Training XGBoost …"):
        t0 = time.time()
        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        xgb_time = time.time() - t0
        xgb_preds = xgb_model.predict(X_test)
        xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

        results["XGBoost"] = {
            "Training time (s)": round(xgb_time, 4),
            "Accuracy":  round(accuracy_score(y_test, xgb_preds), 4),
            "Precision": round(precision_score(y_test, xgb_preds), 4),
            "Recall":    round(recall_score(y_test, xgb_preds), 4),
            "F1 Score":  round(f1_score(y_test, xgb_preds), 4),
            "ROC AUC":   round(roc_auc_score(y_test, xgb_proba), 4),
        }
        models["XGBoost"] = xgb_model

    # ── LightGBM ─────────────────────────────
    with st.spinner("Training LightGBM …"):
        t0 = time.time()
        lgb_model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbosity=-1,
        )
        lgb_model.fit(X_train, y_train)
        lgb_time = time.time() - t0
        lgb_preds = lgb_model.predict(X_test)
        lgb_proba = lgb_model.predict_proba(X_test)[:, 1]

        results["LightGBM"] = {
            "Training time (s)": round(lgb_time, 4),
            "Accuracy":  round(accuracy_score(y_test, lgb_preds), 4),
            "Precision": round(precision_score(y_test, lgb_preds), 4),
            "Recall":    round(recall_score(y_test, lgb_preds), 4),
            "F1 Score":  round(f1_score(y_test, lgb_preds), 4),
            "ROC AUC":   round(roc_auc_score(y_test, lgb_proba), 4),
        }
        models["LightGBM"] = lgb_model

    st.success("✅ Both models trained!")

    # ── Results table ─────────────────────────
    st.subheader("📊 Results Comparison")
    res_df = pd.DataFrame(results).T
    st.dataframe(res_df.style.highlight_max(axis=0, color="#2ecc71")
                              .highlight_min(subset=["Training time (s)"], axis=0, color="#2ecc71"),
                 width="stretch")

    # ── Charts ────────────────────────────────
    st.subheader("📈 Visual Comparison")

    col1, col2 = st.columns(2)

    # Accuracy / F1 bar chart
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
        x = np.arange(len(metrics))
        width = 0.35
        xgb_vals = [results["XGBoost"][m] for m in metrics]
        lgb_vals = [results["LightGBM"][m] for m in metrics]
        ax.bar(x - width/2, xgb_vals, width, label="XGBoost",  color="#3498db")
        ax.bar(x + width/2, lgb_vals, width, label="LightGBM", color="#e67e22")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Classification Metrics")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # ROC curves
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fpr_x, tpr_x, _ = roc_curve(y_test, xgb_proba)
        fpr_l, tpr_l, _ = roc_curve(y_test, lgb_proba)
        ax.plot(fpr_x, tpr_x, label=f"XGBoost (AUC={results['XGBoost']['ROC AUC']:.3f})", color="#3498db")
        ax.plot(fpr_l, tpr_l, label=f"LightGBM (AUC={results['LightGBM']['ROC AUC']:.3f})", color="#e67e22")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)

    # Training time
    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(4, 3))
        bars = ax.bar(results.keys(),
                      [v["Training time (s)"] for v in results.values()],
                      color=["#3498db", "#e67e22"])
        ax.set_ylabel("Seconds")
        ax.set_title("Training Time")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{bar.get_height():.4f}s', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    # Confusion matrices
    with col4:
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        for ax, (name, preds) in zip(axes, [("XGBoost", xgb_preds), ("LightGBM", lgb_preds)]):
            cm = confusion_matrix(y_test, preds)
            im = ax.imshow(cm, cmap="Blues")
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Died", "Survived"], fontsize=8)
            ax.set_yticklabels(["Died", "Survived"], fontsize=8)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)

    # ── Feature Importance ────────────────────
    st.subheader("🏆 Feature Importance")
    col5, col6 = st.columns(2)
    for col, (name, model) in zip([col5, col6], models.items()):
        with col:
            imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
            fig, ax = plt.subplots(figsize=(5, 3.5))
            imp.plot.barh(ax=ax, color="#3498db" if name == "XGBoost" else "#e67e22")
            ax.set_title(f"{name} — Feature Importance")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            st.pyplot(fig)

    # ── Classification Report ─────────────────
    st.subheader("📝 Detailed Classification Reports")
    col7, col8 = st.columns(2)
    with col7:
        st.markdown("**XGBoost**")
        st.text(classification_report(y_test, xgb_preds, target_names=["Died", "Survived"]))
    with col8:
        st.markdown("**LightGBM**")
        st.text(classification_report(y_test, lgb_preds, target_names=["Died", "Survived"]))

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.caption("Built with Streamlit • Titanic Dataset • XGBoost vs LightGBM Benchmark")
