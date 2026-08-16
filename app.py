import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
MODEL_DIR = APP_ROOT / "model"
SAVED_PIPELINES_DIR = MODEL_DIR / "pipelines"
BUNDLED_TEST_DATA_PATH = APP_ROOT / "test_data.csv"

if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))
from model import preprocessing
from model.evaluation_metrics import get_evaluation_metrics

st.set_page_config(page_title="Bank Subscriptions — Model Testing", page_icon="🏦", layout="wide")

MODEL_OPTIONS = {
    "Logistic Regression": "logistic_regression",
    "K-Nearest Neighbors": "knn",
    "Gaussian Naive Bayes": "gnb",
    "Decision Tree": "dt",
    "Random Forest": "rf",
}

# Raw features every model expects
EXPECTED_RAW_COLUMNS = [
    "age", "job", "marital", "education", "default", "balance",
    "housing", "loan", "contact", "day", "month",
    "campaign", "pdays", "previous", "poutcome",
]

JOB_OPTIONS = [
    "admin.", "unknown", "unemployed", "management", "housemaid",
    "entrepreneur", "student", "blue-collar", "self-employed",
    "retired", "technician", "services",
]
MARITAL_OPTIONS = ["married", "divorced", "single"]
EDUCATION_OPTIONS = ["unknown", "primary", "secondary", "tertiary"]
YES_NO_OPTIONS = ["no", "yes"]
CONTACT_OPTIONS = ["unknown", "telephone", "cellular"]
MONTH_OPTIONS = ["jan", "feb", "mar", "apr", "may", "jun",
                  "jul", "aug", "sep", "oct", "nov", "dec"]
POUTCOME_OPTIONS = ["unknown", "other", "failure", "success"]

st.title("🏦 Bank Subscriptions — Model Deployment Testing")
st.caption(
    "Upload a labeled test set (CSV, with a `y` column) to see predictions, "
    "the confusion matrix, classification report, and evaluation metrics for "
    "any of the five trained models."
)

st.session_state.setdefault("test_dataset", None)
st.session_state.setdefault("test_source", None)

# Model selection
selected_model_label = st.selectbox("Choose a model", list(MODEL_OPTIONS.keys()))
selected_model_key = MODEL_OPTIONS[selected_model_label]