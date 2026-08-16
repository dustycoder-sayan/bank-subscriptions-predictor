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
DATASET_INFO_PATH = APP_ROOT / "README.md"

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

@st.cache_data
def load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")

st.title("🏦 Bank Subscriptions — Model Deployment Testing")
st.caption(
    "Upload a labeled test set (CSV, with a `y` column) to see predictions, "
    "the confusion matrix, classification report, and evaluation metrics for "
    "any of the five trained models."
)

st.session_state.setdefault("test_dataset", None)
st.session_state.setdefault("test_source", None)

view = st.radio(
    "View",
    ["🧪 Test Model", "📖 Know about data and Models Used"],
    horizontal=True,
    label_visibility="collapsed",
)
st.divider()

if view == "📖 Know about data and Models Used":
    if DATASET_INFO_PATH.exists():
        st.markdown(load_markdown(DATASET_INFO_PATH))
    else:
        st.warning(
            f"Markdown not available"
        )
    st.stop()

# Model selection
selected_model_label = st.selectbox("Choose a model", list(MODEL_OPTIONS.keys()))
selected_model_key = MODEL_OPTIONS[selected_model_label]

def render_manual_entry_form() -> None:
    with st.form("manual_entry_form"):
        st.markdown("**Enter a value for each feature**")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            job = st.selectbox("Job", JOB_OPTIONS)
            marital = st.selectbox("Marital status", MARITAL_OPTIONS)
            education = st.selectbox("Education", EDUCATION_OPTIONS)
            default = st.selectbox("Has credit in default?", YES_NO_OPTIONS)
        with col2:
            balance = st.number_input("Average yearly balance (€)", value=500, step=50)
            housing = st.selectbox("Has housing loan?", YES_NO_OPTIONS)
            loan = st.selectbox("Has personal loan?", YES_NO_OPTIONS)
            contact = st.selectbox("Contact communication type", CONTACT_OPTIONS)
            day = st.number_input("Last contact day of month", min_value=1, max_value=31, value=15)
        with col3:
            month = st.selectbox("Last contact month", MONTH_OPTIONS)
            campaign = st.number_input("Contacts during this campaign", min_value=1, value=1)
            pdays = st.number_input(
                "Days since last contact (-1 if never contacted)", min_value=-1, value=-1
            )
            previous = st.number_input("Contacts before this campaign", min_value=0, value=0)
            poutcome = st.selectbox("Previous campaign outcome", POUTCOME_OPTIONS)
 
        st.markdown("---")
        duration_input = st.text_input(
            "Call duration in seconds — leave blank if the call hasn't happened yet "
            "(uses the without-duration model); fill in for the with-duration model."
        )
 
        submitted = st.form_submit_button("Predict this row")
 
    if submitted:
        duration_input = duration_input.strip()
        if duration_input:
            try:
                duration_val = float(duration_input)
            except ValueError:
                st.error("Call duration must be a number (or left blank).")
                st.stop()
        else:
            duration_val = None  # blank -> without_duration variant
 
        manually_entered_data = {
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "balance": balance, "housing": housing, "loan": loan,
            "contact": contact, "day": day, "month": month, "campaign": campaign,
            "pdays": pdays, "previous": previous, "poutcome": poutcome,
        }
        if duration_val is not None:
            manually_entered_data["duration"] = duration_val
 
        st.session_state["test_dataset"] = pd.DataFrame([manually_entered_data])
        st.session_state["test_source"] = "Manually entered data"
        st.rerun()