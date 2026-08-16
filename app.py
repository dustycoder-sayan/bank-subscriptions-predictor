import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

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

if st.session_state["test_dataset"] is None:
    input_mode = st.radio(
        "How would you like to provide data?",
        ["Upload a CSV", "Enter values manually"],
        horizontal=True,
    )
 
    if input_mode == "Upload a CSV":
        left, right = st.columns(2)
 
        with left:
            uploaded_file = st.file_uploader("Upload a test CSV", type=["csv"])
            if uploaded_file is not None:
                try:
                    df_uploaded = pd.read_csv(uploaded_file)
                except Exception as e:
                    st.error(f"Couldn't read input file: {e}")
                    st.stop()
                st.session_state["test_dataset"] = df_uploaded
                st.session_state["test_source"] = f"Uploaded file: {uploaded_file.name}"
                st.rerun()
 
        with right:
            st.write("")  # vertical alignment nudge
            st.write("")
            if BUNDLED_TEST_DATA_PATH.exists():
                if st.button("Use bundled test_data.csv instead"):
                    st.session_state["test_dataset"] = pd.read_csv(BUNDLED_TEST_DATA_PATH)
                    st.session_state["test_source"] = "Bundled test_data.csv"
                    st.rerun()
    else:
        render_manual_entry_form()
else:
    top_left, top_right = st.columns([4, 1])
    with top_left:
        st.success(
            f"Using data from **{st.session_state['test_source']}** "
            f"({len(st.session_state['test_dataset'])} row(s)). "
            "This stays in place across model changes until you remove it."
        )
    with top_right:
        if st.button("🗑️ Remove dataset"):
            st.session_state["test_dataset"] = None
            st.session_state["test_source"] = None
            st.rerun()

    with st.expander("Preview data", expanded=False):
        st.dataframe(st.session_state["test_dataset"])

# Predict + evaluate selected model
if st.session_state["test_dataset"] is not None:
    df_input = st.session_state["test_dataset"].copy()

    has_duration = "duration" in df_input.columns
    has_target = "y" in df_input.columns

    missing_cols = [c for c in EXPECTED_RAW_COLUMNS if c not in df_input.columns]
    if missing_cols:
        st.error(
            "This dataset is missing columns the model needs to run: "
            f"`{', '.join(missing_cols)}`. Please include all required features."
        )
        st.stop()

    variant_dir = "with_duration" if has_duration else "without_duration"
    model_path = SAVED_PIPELINES_DIR / variant_dir / f"{selected_model_key}_pipeline.joblib"

    st.info(
        f"Detected **{'with' if has_duration else 'without'} duration** data "
        f"→ using the `{selected_model_label}` model trained {'with' if has_duration else 'without'} "
        f"duration"
    )

    if not model_path.exists():
        st.error(
            f"To the devloper: Couldn't find a saved model at `{model_path}`."
            " Make sure this model has been trained and exported for this data variant."
        )
        st.stop()

    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]

    X_for_prediction = df_input.drop(columns=["y"], errors="ignore")

    try:
        proba = pipeline.predict_proba(X_for_prediction)[:, -1]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    pred = (proba >= threshold).astype(int)
    pred_labels = np.where(pred == 1, "yes", "no")

    # Predictions
    st.subheader("Predictions")
    results_df = df_input.copy()
    results_df["predicted_probability"] = proba.round(4)
    results_df["predicted_outcome"] = pred_labels
    st.dataframe(results_df, use_container_width=True)

    st.download_button(
        "⬇️ Download predictions as CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_model_key}_{variant_dir}_predictions.csv",
        mime="text/csv",
    )

    # Evaluation (only if this is a labeled - test dataset)
    if has_target:
        y_true_raw = df_input["y"]
        if pd.api.types.is_numeric_dtype(y_true_raw):
            y_true = y_true_raw.astype(int)
        else:
            y_true = y_true_raw.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})

        if y_true.isna().any():
            st.warning(
                "Some values in the `y` column couldn't be interpreted as "
                "\"yes\"/\"no\" — evaluation metrics may be unreliable."
            )

        metrics, report_dict, cm_fig = get_evaluation_metrics(
            selected_model_label, y_true, proba, threshold=threshold
        )

        # Deepcopy fig to display on page
        safe_fig = Figure(figsize=(4, 3))
        safe_ax = safe_fig.subplots()

        # Copy all elements from cm_fig
        for ax in cm_fig.get_axes():
            # Transfer the image (the confusion matrix heatmap grid)
            for im in ax.get_images():
                safe_ax.imshow(im.get_array(), cmap=im.get_cmap(), extent=im.get_extent())
            
            # Matrix cell numbers
            for text in ax.texts:
                safe_ax.text(
                    text.get_position()[0], text.get_position()[1], 
                    text.get_text(), ha=text.get_ha(), va=text.get_va(), 
                    color=text.get_color()
                )
            
            # Titles and Ticks
            safe_ax.set_title(ax.get_title())
            safe_ax.set_xticks(ax.get_xticks())
            safe_ax.set_yticks(ax.get_yticks())
            safe_ax.set_xticklabels([label.get_text() for label in ax.get_xticklabels()])
            safe_ax.set_yticklabels([label.get_text() for label in ax.get_yticklabels()])
            safe_ax.set_xlabel(ax.get_xlabel())
            safe_ax.set_ylabel(ax.get_ylabel())

        safe_fig.tight_layout()

        plt.close(cm_fig)

        st.subheader("Confusion Matrix")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.pyplot(cm_fig)

        st.subheader("Classification Report")
        report_df = pd.DataFrame(report_dict).T
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

        st.subheader("Evaluation Metrics")
        st.dataframe(pd.DataFrame([metrics]), use_container_width=True)
    else:
        st.caption(
            "No `y` column detected in this dataset — showing predictions only. "
            "Upload a labeled test set to also see evaluation metrics."
        )