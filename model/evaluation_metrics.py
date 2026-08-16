"""
Shared evaluation metrics helper.

This is imported by BOTH the notebook (during model development/comparison) 
and app.py (when scoring a user-uploaded test set) — a single source of truth 
for how metrics are computed, so the numbers reported during training match what the app shows.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    matthews_corrcoef,
    classification_report,
)


def get_evaluation_metrics(model_name: str, y_true, y_score, threshold: float = 0.5):
    """
    Derive predictions at a given threshold and calculate core classification
    performance metrics, a classification report, and a confusion matrix figure.

    Parameters
    ----------
    model_name : str
        Name of the model being evaluated (used in the confusion matrix title).
    y_true : array-like of shape (n_samples,)
        Ground truth target values, encoded as 0 ("no") / 1 ("yes").
    y_score : array-like of shape (n_samples,)
        Probability estimates for the positive class ("yes"), typically from
        `pipeline.predict_proba(X)[:, -1]`.
    threshold : float, default=0.5
        Probability cutoff used to derive y_pred from y_score. Pass the model's
        tuned threshold (stored alongside the pipeline in its joblib bundle)
        rather than leaving this at the default.

    Returns
    -------
    metrics : dict
        Accuracy, AUC, precision, recall, F1, MCC, and the threshold used.
    report_dict : dict
        Output of `sklearn.metrics.classification_report(..., output_dict=True)`,
        ready to be turned into a DataFrame via `pd.DataFrame(report_dict).T`.
    fig : matplotlib.figure.Figure
        Confusion matrix heatmap, ready to be rendered via `st.pyplot(fig)`.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    # --- Confusion matrix figure ---
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no", "yes"])
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}\n(threshold = {threshold:.3f})")
    fig.tight_layout()

    # --- Core metrics ---
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tn + fp + fn + tp) if (tn + fp + fn + tp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    # AUC needs both classes present in y_true — guard against a degenerate
    # single-class batch (e.g. a manually entered single row).
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    mcc = matthews_corrcoef(y_true, y_pred)

    metrics = {
        "accuracy": accuracy,
        "auc_score": auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mcc": mcc,
        "threshold": threshold,
    }

    # --- Classification report ---
    report_dict = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["no", "yes"],
        output_dict=True, zero_division=0,
    )

    return metrics, report_dict, fig
