"""
src/preprocessing.py
====================
Data loading, cleaning, feature engineering, and sklearn preprocessor construction
for the Telco Customer Churn prediction project.

All functions are pure (no global state). The preprocessor must be fit on training
data only and then applied to the test set — this is enforced by the API design:
``build_preprocessor`` takes an already-split X_train.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Column taxonomy — single source of truth for the entire project
# ---------------------------------------------------------------------------

# Columns that are dropped early (ID-like, not predictive)
DROP_COLS: list[str] = ["customerID"]

# Target column
TARGET_COL: str = "Churn"

# Numeric features BEFORE engineering (used for scaling)
NUMERIC_RAW: list[str] = ["tenure", "MonthlyCharges", "TotalCharges"]

# Binary Yes/No categorical columns (two-level — binary encode as 0/1)
BINARY_COLS: list[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Nominal multi-category columns (3+ levels — one-hot encode)
NOMINAL_COLS: list[str] = ["InternetService", "Contract", "PaymentMethod"]

# Engineered numeric features (added later, also scaled)
ENGINEERED_NUMERIC: list[str] = [
    "Avg_Monthly_Spend",
    "Total_Services_Subscribed",
]

# Engineered ordinal / flag features (treat as numeric for simplicity)
ENGINEERED_FLAGS: list[str] = [
    "Tenure_Group",
    "Charge_Category",
    "Is_Long_Term_Customer",
    "Is_High_Value_Customer",
]


# ---------------------------------------------------------------------------
# 1. Load & Clean
# ---------------------------------------------------------------------------


def load_and_clean(path: str) -> pd.DataFrame:
    """Load the Telco churn CSV, fix dtypes, and handle known data quality issues.

    Parameters
    ----------
    path : str
        Absolute or relative path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with correct dtypes, no duplicates, and no
        propagated NaN values.

    Notes
    -----
    Known issues addressed:
    - ``TotalCharges`` is stored as object/string in the raw dataset because
      some rows with ``tenure == 0`` have a blank string instead of "0".
      These are imputed as 0 (they have not yet made any payments).
      Any *other* non-numeric TotalCharges values are coerced to NaN and
      then dropped (they represent genuine data errors, not a structural pattern).
    - ``SeniorCitizen`` is stored as 0/1 int but is semantically categorical.
      It is converted to "Yes"/"No" strings so it flows through the same
      binary-encode path as the other Yes/No features.
    - Duplicate rows (by customerID) are dropped, keeping the first occurrence.
    """
    df = pd.read_csv(path)

    # --- Drop ID column ---
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # --- Fix TotalCharges: blank → 0 for tenure-0 rows, NaN elsewhere → drop ---
    if "TotalCharges" in df.columns:
        # Replace blank strings with NaN first
        df["TotalCharges"] = df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True)

        # For tenure==0 rows, NaN TotalCharges makes sense → impute as 0
        tenure_zero_mask = df["tenure"] == 0
        df.loc[tenure_zero_mask & df["TotalCharges"].isna(), "TotalCharges"] = 0.0

        # Convert to numeric; any remaining non-numeric → NaN → drop
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        n_before = len(df)
        df = df.dropna(subset=["TotalCharges"])
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(
                f"[preprocessing] Dropped {n_dropped} rows with non-numeric "
                "TotalCharges (data errors, not tenure-0 blanks)."
            )

    # --- Deduplicate (safety net — dataset should be unique by customerID) ---
    n_before = len(df)
    df = df.drop_duplicates()
    n_dup = n_before - len(df)
    if n_dup > 0:
        print(f"[preprocessing] Dropped {n_dup} exact duplicate rows.")

    # --- SeniorCitizen: int → categorical string ---
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # --- Target encode: Churn Yes/No → 1/0 ---
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)

    # --- Normalise "No internet service" / "No phone service" → "No" ---
    # These are redundant sub-categories; the internet/phone columns already
    # capture the parent service.  Collapsing reduces cardinality and avoids
    # spurious splits in tree models.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace(
            {"No internet service": "No", "No phone service": "No"}
        )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-driven engineered features and append them to *df*.

    All transformations are non-destructive: original columns are preserved.
    Every engineered feature's relationship to churn is documented in the
    notebook; the logic here is the authoritative implementation.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (output of ``load_and_clean``).

    Returns
    -------
    pd.DataFrame
        Input DataFrame extended with the following new columns:

        Tenure_Group : int (0–3)
            Ordinal bucket: 0 = 0–12 months, 1 = 13–24, 2 = 25–48, 3 = 49+.
            Chosen because churn rate drops sharply after 24 months and again
            after 48 months — these breakpoints reflect natural loyalty stages.

        Charge_Category : int (0–2)
            Ordinal bucket for MonthlyCharges: 0 = low (<40), 1 = mid (40–70),
            2 = high (>70). Mirrors the churn-rate jump observed in EDA.

        Avg_Monthly_Spend : float
            TotalCharges / max(tenure, 1). Represents actual average monthly
            revenue per customer. Guarded against divide-by-zero: tenure=0
            customers use tenure=1 as the denominator (they are one month in).

        Is_Long_Term_Customer : int (0/1)
            1 if tenure >= 24 months, else 0. Threshold chosen because EDA
            shows churn rate drops below 15% after 24 months.

        Is_High_Value_Customer : int (0/1)
            1 if MonthlyCharges >= 70, else 0. High-spend customers who churn
            represent significant revenue loss; flagging them enables targeted
            retention strategies.

        Total_Services_Subscribed : int (0–8)
            Count of add-on services: PhoneService, MultipleLines,
            OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
            StreamingTV, StreamingMovies. More services → higher switching
            cost → lower churn. This is a genuinely predictive feature.
    """
    df = df.copy()

    # --- Tenure_Group (ordinal bins matching natural loyalty stages) ---
    tenure_bins = [-1, 12, 24, 48, 9999]
    tenure_labels = [0, 1, 2, 3]  # ordinal integers, not strings
    df["Tenure_Group"] = pd.cut(
        df["tenure"], bins=tenure_bins, labels=tenure_labels
    ).astype(int)

    # --- Charge_Category (ordinal: low / mid / high) ---
    charge_bins = [-1, 40, 70, 9999]
    charge_labels = [0, 1, 2]
    df["Charge_Category"] = pd.cut(
        df["MonthlyCharges"], bins=charge_bins, labels=charge_labels
    ).astype(int)

    # --- Avg_Monthly_Spend (guard divide-by-zero for tenure=0) ---
    safe_tenure = df["tenure"].clip(lower=1)
    df["Avg_Monthly_Spend"] = df["TotalCharges"] / safe_tenure

    # --- Is_Long_Term_Customer (binary flag, threshold = 24 months) ---
    df["Is_Long_Term_Customer"] = (df["tenure"] >= 24).astype(int)

    # --- Is_High_Value_Customer (binary flag, threshold = $70/month) ---
    df["Is_High_Value_Customer"] = (df["MonthlyCharges"] >= 70).astype(int)

    # --- Total_Services_Subscribed ---
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    # Map Yes→1, No→0 in a temp view; sum across columns
    service_df = df[service_cols].apply(lambda col: (col == "Yes").astype(int))
    df["Total_Services_Subscribed"] = service_df.sum(axis=1)

    return df


# ---------------------------------------------------------------------------
# 3. Build Preprocessor (fit on train only — leakage prevention)
# ---------------------------------------------------------------------------


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Construct and fit a ColumnTransformer on *X_train*.

    ⚠️  LEAKAGE PREVENTION: This function MUST receive only the training
    partition.  The returned transformer is then used with ``.transform()``
    (not ``.fit_transform()``) on the test set, ensuring that no statistics
    (means, category vocabularies) computed from test data influence the model.

    Encoding strategy:
    - Numeric (raw + engineered) → StandardScaler
      Required for Logistic Regression; tree models are scale-invariant but a
      single preprocessing branch keeps the pipeline simple and reproducible.
    - Binary Yes/No columns → binary map (drop='if_binary' OHE)
      A two-level column has no ordinal relationship issue; one dummy suffices.
    - Nominal multi-category → OneHotEncoder (drop='first' to avoid multicollinearity)
      Correct for columns like InternetService (DSL / Fiber / No) where no
      ordinal relationship exists.  LabelEncoder would impose an arbitrary
      numeric order that tree splitters and linear models would misinterpret.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix of the training partition (target column excluded).

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Fitted transformer ready to be applied to train and test sets.
    """
    # Identify which columns are present (engineered features may not always exist)
    present_numeric = [
        c
        for c in NUMERIC_RAW + ENGINEERED_NUMERIC + ENGINEERED_FLAGS
        if c in X_train.columns
    ]
    present_binary = [c for c in BINARY_COLS if c in X_train.columns]
    # Add SeniorCitizen to binary cols if present
    if "SeniorCitizen" in X_train.columns and "SeniorCitizen" not in present_binary:
        present_binary.append("SeniorCitizen")
    present_nominal = [c for c in NOMINAL_COLS if c in X_train.columns]

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    binary_transformer = Pipeline(
        steps=[
            (
                "ohe",
                OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore"),
            )
        ]
    )

    nominal_transformer = Pipeline(
        steps=[
            (
                "ohe",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, present_numeric),
            ("bin", binary_transformer, present_binary),
            ("nom", nominal_transformer, present_nominal),
        ],
        remainder="drop",  # drop any unlisted columns (e.g. engineered ordinals already included)
    )

    preprocessor.fit(X_train)
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract human-readable feature names from a fitted ColumnTransformer.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        A fitted ColumnTransformer (output of ``build_preprocessor``).

    Returns
    -------
    list[str]
        Ordered list of output feature names matching the transformed matrix columns.
    """
    feature_names: list[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            feature_names.extend(transformer.get_feature_names_out(cols).tolist())
        elif hasattr(transformer, "steps"):
            last_step = transformer.steps[-1][1]
            if hasattr(last_step, "get_feature_names_out"):
                feature_names.extend(last_step.get_feature_names_out(cols).tolist())
            else:
                feature_names.extend(cols if isinstance(cols, list) else [cols])
        else:
            feature_names.extend(cols if isinstance(cols, list) else [cols])
    return feature_names
