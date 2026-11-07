import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

import sys
import sklearn
import joblib
import streamlit as st

st.sidebar.subheader("Environment Information")

st.sidebar.write("Python:", sys.version)
st.sidebar.write("Scikit-learn:", sklearn.__version__)
st.sidebar.write("Joblib:", joblib.__version__)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

FEATURE_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

MODEL_FILES = {
    "Logistic Regression (Baseline)": "logistic_regression.joblib",
    "Decision Tree (Baseline)": "decision_tree.joblib",
    "K-Nearest Neighbours (Baseline)": "knn.joblib",
    "Gaussian Naive Bayes (Baseline)": "gaussian_nb.joblib",
    "Support Vector Machine (Baseline)": "svm.joblib",
    "Decision Tree (Tuned)": "tuned_decision_tree.joblib",
    "Logistic Regression (Tuned)": "tuned_logistic_regression.joblib",
}

st.set_page_config(
    page_title="Online Shopping Purchase Prediction",
    page_icon="🛒",
    layout="wide",
)

# ---------------------------------------------------------------------
# Model loading: models are loaded from disk; no training is performed.
# ---------------------------------------------------------------------
@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def convert_target_to_binary(target: pd.Series) -> pd.Series:
    """Convert Revenue values such as True/False or 1/0 to integers."""
    mapping = {
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
    }
    converted = target.astype(str).str.strip().str.lower().map(mapping)

    if converted.isna().any():
        bad_values = target[converted.isna()].astype(str).unique().tolist()
        raise ValueError(
            "Revenue must contain only True/False or 1/0 values. "
            f"Unrecognised values: {bad_values[:10]}"
        )

    return converted.astype(int)


def get_prediction_scores(model, features: pd.DataFrame):
    """Return scores for the positive class, for ROC-AUC calculation."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        classes = list(model.classes_)
        positive_index = classes.index(1)
        return probabilities[:, positive_index]

    if hasattr(model, "decision_function"):
        return model.decision_function(features)

    return None


# ---------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------
st.title("🛒 Online Shopping Purchase Intention Prediction")
st.write(
    "This application predicts whether an online shopping session results "
    "in a purchase (`Revenue = True`) or not (`Revenue = False`). "
    "Select a pre-trained machine learning model and upload a labelled test CSV."
)

st.info(
    "The uploaded CSV must contain all 17 input features and the actual "
    "`Revenue` target column. The application loads saved models and does "
    "not retrain them."
)

with st.expander("Required CSV columns"):
    st.code(", ".join(FEATURE_COLUMNS + ["Revenue"]), language="text")

uploaded_file = st.file_uploader(
    "Upload test CSV",
    type=["csv"],
    help="Upload test_data.csv generated from the assignment test split.",
)

model_label = st.selectbox(
    "Select a classification model",
    options=list(MODEL_FILES.keys()),
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        missing_columns = [
            column for column in FEATURE_COLUMNS + ["Revenue"]
            if column not in data.columns
        ]

        if missing_columns:
            st.error(
                "The uploaded CSV is missing these required columns: "
                + ", ".join(missing_columns)
            )
            st.stop()

        if data.empty:
            st.error("The uploaded CSV contains no rows.")
            st.stop()

        # Keep only the required columns and preserve the feature order.
        X_input = data[FEATURE_COLUMNS].copy()
        y_true = convert_target_to_binary(data["Revenue"])

        # Weekend was used as a numeric binary feature during training.
        X_input["Weekend"] = X_input["Weekend"].map(
            lambda value: int(value)
            if isinstance(value, (bool, np.bool_))
            else value
        )

        model_filename = MODEL_FILES[model_label]
        model_path = MODELS_DIR / model_filename

        if not model_path.exists():
            st.error(
                f"Model file not found: {model_path}. "
                "Make sure the saved .joblib files are inside the models folder."
            )
            st.stop()

        model = load_model(str(model_path))

        if st.button("Evaluate Model", type="primary"):
            try:
                predictions = model.predict(X_input).astype(int)
                prediction_scores = get_prediction_scores(model, X_input)

                accuracy = accuracy_score(y_true, predictions)
                precision = precision_score(
                    y_true, predictions, zero_division=0
                )
                recall = recall_score(y_true, predictions, zero_division=0)
                f1 = f1_score(y_true, predictions, zero_division=0)
                mcc = matthews_corrcoef(y_true, predictions)

                auc = (
                    roc_auc_score(y_true, prediction_scores)
                    if prediction_scores is not None and y_true.nunique() == 2
                    else None
                )

                st.subheader("Evaluation results")
                st.caption(f"Selected model: {model_label}")

                metric_columns = st.columns(6)
                metric_values = [
                    ("Accuracy", accuracy),
                    ("AUC", auc),
                    ("Precision", precision),
                    ("Recall", recall),
                    ("F1-score", f1),
                    ("MCC", mcc),
                ]

                for column, (label, value) in zip(metric_columns, metric_values):
                    column.metric(
                        label,
                        f"{value:.4f}" if value is not None else "N/A",
                    )

                left, right = st.columns(2)

                with left:
                    st.subheader("Confusion matrix")
                    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
                    cm_df = pd.DataFrame(
                        cm,
                        index=["Actual: No Purchase", "Actual: Purchase"],
                        columns=["Predicted: No Purchase", "Predicted: Purchase"],
                    )
                    st.dataframe(cm_df, use_container_width=True)

                with right:
                    st.subheader("Classification report")
                    report = classification_report(
                        y_true,
                        predictions,
                        labels=[0, 1],
                        target_names=["No Purchase (0)", "Purchase (1)"],
                        output_dict=True,
                        zero_division=0,
                    )
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(
                        report_df.round(4),
                        use_container_width=True,
                    )

                st.subheader("Predictions")
                results = data.copy()
                results["Actual_Revenue"] = y_true.values
                results["Predicted_Revenue"] = predictions
                results["Predicted_Label"] = np.where(
                    predictions == 1, "Purchase", "No Purchase"
                )

                st.dataframe(results.head(20), use_container_width=True)

                csv_bytes = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions CSV",
                    data=csv_bytes,
                    file_name="purchase_predictions.csv",
                    mime="text/csv",
                )

            except Exception as error:
                st.error(f"Unable to evaluate the selected model: {error}")

    except Exception as error:
        st.error(f"Unable to read or validate the uploaded CSV: {error}")

else:
    st.warning("Please upload a labelled test CSV to evaluate a model.")

st.divider()
st.caption(
    "Academic machine learning assignment project | "
    "Online Shoppers Purchasing Intention Dataset (UCI)"
)
