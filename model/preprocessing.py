"""
Shared preprocessing components for the deployable model pipelines.

joblib records the exact module path of every custom class or function a
pickled object depends on. Anything defined inline in a notebook cell gets
recorded under the module `__main__` of that notebook's kernel — which is not
resolvable from any other process (e.g: The streamlit application), so unpickling it
elsewhere raises `AttributeError: Can't get attribute ... on <module '__main__'>`.
Defining these here, in a real importable module, avoids that entirely.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder

DROP_ALWAYS = ["y", "day", "default"]
BINARY_COLS = ["housing", "loan"]
ORDINAL_COLS = ["education"]
EDUCATION_ORDER = ["unknown", "primary", "secondary", "tertiary"]
NOMINAL_COLS = ["job", "marital", "contact", "poutcome", "month"]


def clean_and_engineer(df: pd.DataFrame, with_duration: bool = True) -> pd.DataFrame:
    """Drops unwanted columns and engineers features. Every column access is
    existence-checked first, so a missing column is skipped rather than raising."""
    df_clean = df.copy()
    df_clean = df_clean.drop(columns=DROP_ALWAYS, errors="ignore")
    if not with_duration:
        df_clean = df_clean.drop(columns=["duration"], errors="ignore")

    if "pdays" in df_clean.columns:
        df_clean["was_previously_contacted"] = (df_clean["pdays"] != -1).astype(int)
        df_clean["pdays"] = df_clean["pdays"].replace(-1, 0)
    if "balance" in df_clean.columns:
        df_clean["balance"] = np.sign(df_clean["balance"]) * np.log1p(np.abs(df_clean["balance"]))

    return df_clean


class SafeEncoder(BaseEstimator, TransformerMixin):
    """Builds a ColumnTransformer at fit-time using only the expected columns
    that are actually present, so missing columns never crash it. Unseen
    categories at transform-time are handled gracefully, not raised."""

    def __init__(self, binary_cols=BINARY_COLS, ordinal_cols=ORDINAL_COLS,
                 education_order=EDUCATION_ORDER, nominal_cols=NOMINAL_COLS):
        self.binary_cols = binary_cols
        self.ordinal_cols = ordinal_cols
        self.education_order = education_order
        self.nominal_cols = nominal_cols

    def fit(self, X: pd.DataFrame, y=None):
        cols = set(X.columns)
        present_binary = [c for c in self.binary_cols if c in cols]
        present_ordinal = [c for c in self.ordinal_cols if c in cols]
        present_nominal = [c for c in self.nominal_cols if c in cols]

        transformers = []
        if present_binary:
            transformers.append((
                "binary",
                OrdinalEncoder(
                    categories=[["no", "yes"]] * len(present_binary),
                    handle_unknown="use_encoded_value", unknown_value=-1,
                ),
                present_binary,
            ))
        if present_ordinal:
            transformers.append((
                "ordinal",
                OrdinalEncoder(
                    categories=[self.education_order] * len(present_ordinal),
                    handle_unknown="use_encoded_value", unknown_value=-1,
                ),
                present_ordinal,
            ))
        if present_nominal:
            transformers.append((
                "nominal",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                present_nominal,
            ))

        self.column_transformer_ = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        self.column_transformer_.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame):
        out = self.column_transformer_.transform(X)
        cols = self.column_transformer_.get_feature_names_out()
        return pd.DataFrame(out, columns=cols, index=X.index)


def build_common_preprocessor(with_duration: bool = True) -> Pipeline:
    """Returns a preprocessing Pipeline (cleaning + safe encoding) that can be
    reused as the 'preprocessor' step across all deployable model pipelines."""
    return Pipeline(steps=[
        ("cleaning", FunctionTransformer(
            func=clean_and_engineer,
            kw_args={"with_duration": with_duration},
            validate=False,
        )),
        ("encoding", SafeEncoder()),
    ])