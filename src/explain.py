"""
src/explain.py
==============
SHAP computation, customer risk prediction, and business recommendation
generation for the Telco Customer Churn prediction project.

Design notes
------------
- ``predict_customer()`` loads saved artifacts (preprocessor + model) and
  applies the tuned classification threshold.  It never re-fits anything —
  re-fitting inside a prediction function would be a critical production bug.
- ``business_recommendations()`` is conditioned on the actual SHAP-driven
  top contributors for the specific customer, not a static lookup table keyed
  only on the overall churn prediction.  This makes recommendations
  actionable and personalised.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------


def compute_shap(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    max_background_samples: int = 100,
) -> Tuple[Any, np.ndarray]:
    """Compute SHAP values for *X_test* using an appropriate explainer.

    Explainer selection strategy:
    - Tree-based models (RF, GB, XGB, CatBoost) → ``TreeExplainer``
      (exact, fast, no sampling required).
    - Linear models (LR) → ``LinearExplainer``.
    - Fallback → ``KernelExplainer`` with a subsampled background set
      (model-agnostic but slow; the background is capped to avoid OOM).

    Parameters
    ----------
    model : fitted sklearn/XGB/CatBoost estimator
    X_train : np.ndarray
        Training features (used as background for KernelExplainer or for
        computing expected values in LinearExplainer).
    X_test : np.ndarray
        Test features to explain.
    feature_names : list[str]
        Column names matching X_train/X_test columns.
    max_background_samples : int
        Maximum rows of X_train used as KernelExplainer background.

    Returns
    -------
    (explainer, shap_values)
        ``shap_values`` has shape (n_test, n_features) for binary classification
        (values for the positive class).
    """
    model_type = type(model).__name__

    tree_types = {
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "DecisionTreeClassifier",
        "XGBClassifier",
        "CatBoostClassifier",
        "ExtraTreesClassifier",
    }

    if model_type in tree_types:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)
        # For binary classifiers, shap_values may return [neg_class, pos_class]
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]
    elif model_type == "LogisticRegression":
        background = shap.maskers.Independent(X_train, max_samples=max_background_samples)
        explainer = shap.LinearExplainer(model, background)
        shap_vals = explainer.shap_values(X_test)
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]
    else:
        # Fallback: KernelExplainer (slow but universal)
        background = X_train[
            np.random.choice(X_train.shape[0], max_background_samples, replace=False)
        ]
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_vals = explainer.shap_values(X_test)
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]

    return explainer, shap_vals


def plot_shap_summary(
    shap_values: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    max_display: int = 20,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Render a SHAP beeswarm summary plot.

    Parameters
    ----------
    shap_values : np.ndarray
        Shape (n_samples, n_features).
    X_test : np.ndarray
        Feature values (used for colouring the beeswarm dots).
    feature_names : list[str]
    max_display : int
        Maximum number of features to display (sorted by mean |SHAP|).
    figsize : tuple
        Figure size passed to plt.figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title("SHAP Summary — Global Feature Importance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_shap_waterfall(
    explainer: Any,
    shap_values: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    sample_idx: int,
    title: str = "SHAP Waterfall",
    figsize: Tuple[int, int] = (10, 7),
) -> plt.Figure:
    """Render a SHAP waterfall plot for a single observation.

    Parameters
    ----------
    explainer : shap.Explainer
        Fitted explainer (used for expected_value).
    shap_values : np.ndarray
        Shape (n_samples, n_features).
    X_test : np.ndarray
        Feature values for all test observations.
    feature_names : list[str]
    sample_idx : int
        Index of the observation to explain.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Build an Explanation object for the waterfall plot API
    if hasattr(explainer, "expected_value"):
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = ev[-1] if len(ev) == 2 else ev[0]
    else:
        ev = float(np.mean(shap_values))

    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=ev,
        data=X_test[sample_idx],
        feature_names=feature_names,
    )

    fig = plt.figure(figsize=figsize)
    shap.waterfall_plot(explanation, max_display=15, show=False)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Customer risk prediction
# ---------------------------------------------------------------------------


def predict_customer(
    customer_dict: Dict[str, Any],
    preprocessor: Any,
    model: Any,
    threshold: float,
    explainer: Any,
    feature_names: List[str],
    top_n_features: int = 5,
) -> Dict[str, Any]:
    """End-to-end churn prediction for a single raw customer record.

    ⚠️  This function uses the saved preprocessor and model artifacts — it
    does NOT re-fit anything.  Passing an unfitted preprocessor or model
    will raise a sklearn NotFittedError, which is the correct fail-fast
    behaviour.

    Parameters
    ----------
    customer_dict : dict
        Raw customer features as a flat dictionary (same keys as the raw CSV,
        *without* customerID, *without* Churn).  The preprocessor handles all
        encoding and scaling.
    preprocessor : fitted ColumnTransformer
        Loaded from ``models/preprocessor.joblib``.
    model : fitted estimator
        Loaded from ``models/best_model.joblib``.
    threshold : float
        Tuned classification threshold (from Section 14, NOT 0.5).
    explainer : shap.Explainer
        Fitted SHAP explainer for the model.
    feature_names : list[str]
        Ordered feature names matching the preprocessor's output.
    top_n_features : int
        Number of top SHAP contributors to return.

    Returns
    -------
    dict with keys:
        prediction : int
            1 = likely to churn, 0 = likely to stay.
        churn_probability : float
            Raw model probability for class 1 (churn), in [0, 1].
        risk_score : int
            0–100 monotonic in churn_probability.  Mapping:
            risk_score = round(churn_probability * 100).
        confidence : float
            Absolute distance of churn_probability from the tuned threshold.
            Higher → the model is more certain about its prediction.
        top_features : list[dict]
            Top ``top_n_features`` SHAP contributors, each dict having:
            ``feature`` (str), ``value`` (raw input value), ``shap_value``
            (signed float, positive = pushes toward churn).
    """
    from src.preprocessing import engineer_features

    # Step 1: Convert dict → DataFrame and run feature engineering
    raw_df = pd.DataFrame([customer_dict])
    
    # Fix SeniorCitizen 0/1 to No/Yes if it was passed as integer
    if "SeniorCitizen" in raw_df.columns:
        raw_df["SeniorCitizen"] = raw_df["SeniorCitizen"].map({0: "No", 1: "Yes", "0": "No", "1": "Yes"}).fillna(raw_df["SeniorCitizen"])

    engineered_df = engineer_features(raw_df)

    # Step 2: Preprocess (transform only — no fitting)
    X_processed = preprocessor.transform(engineered_df)

    # Step 3: Predict probability
    churn_prob = float(model.predict_proba(X_processed)[0, 1])

    # Step 4: Apply tuned threshold
    prediction = int(churn_prob >= threshold)

    # Step 5: Risk score (monotonic 0–100 mapping)
    risk_score = round(churn_prob * 100)

    # Step 6: Confidence (distance from threshold)
    confidence = round(abs(churn_prob - threshold), 4)

    # Step 7: SHAP-based top feature contributors
    model_type = type(model).__name__
    tree_types = {
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "DecisionTreeClassifier",
        "XGBClassifier",
        "CatBoostClassifier",
    }

    sv = explainer.shap_values(X_processed)
    
    if isinstance(sv, list) and len(sv) == 2:
        sv = sv[1]
    elif isinstance(sv, np.ndarray) and sv.ndim == 3 and sv.shape[2] == 2:
        sv = sv[:, :, 1]
        
    instance_shap = sv[0]

    top_indices = np.argsort(np.abs(instance_shap))[::-1][:top_n_features]
    top_features = []
    for idx in top_indices:
        fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        top_features.append(
            {
                "feature": fname,
                "value": float(X_processed[0, idx]),
                "shap_value": round(float(instance_shap[idx]), 4),
            }
        )

    return {
        "prediction": prediction,
        "churn_probability": round(churn_prob, 4),
        "risk_score": risk_score,
        "confidence": confidence,
        "top_features": top_features,
    }


# ---------------------------------------------------------------------------
# Business recommendations (conditioned on SHAP contributors)
# ---------------------------------------------------------------------------

# Feature-to-recommendation mapping: each entry maps a feature-name substring
# to a (recommendation, rationale) tuple.  Matched against top SHAP features.
_RECOMMENDATION_RULES: List[Tuple[str, str, str]] = [
    # (feature_substring, recommendation, brief_rationale)
    (
        "Contract",
        "Offer a 12- or 24-month contract incentive (e.g. first month free, "
        "locked-in rate, or loyalty discount).",
        "Month-to-month customers churn at 3–4× the rate of annual subscribers.",
    ),
    (
        "tenure",
        "Enrol customer in a loyalty milestone program (e.g. 6-month reward, "
        "free upgrade at 12 months).",
        "Early-tenure customers are at the highest churn risk; milestone rewards "
        "increase switching cost and perceived value.",
    ),
    (
        "Tenure_Group",
        "Enrol customer in a loyalty milestone program to increase switching cost.",
        "Short-tenure group customers churn significantly more often.",
    ),
    (
        "MonthlyCharges",
        "Review the customer's plan for right-sizing or offer a discounted bundle.",
        "High monthly charges are a top churn driver; a targeted price concession "
        "can retain a high-value customer at positive margin.",
    ),
    (
        "InternetService",
        "Highlight speed, reliability, or offer a fibre upgrade trial.",
        "Internet service type is strongly associated with churn; fibre customers "
        "on expensive plans churn when they perceive poor value.",
    ),
    (
        "TechSupport",
        "Offer a complimentary 3-month tech support trial.",
        "Lack of tech support is a top-5 churn driver; customers who experience "
        "unresolved technical issues leave.",
    ),
    (
        "OnlineSecurity",
        "Offer a free 3-month online security add-on trial.",
        "Security is a low-cost upsell that increases perceived value and "
        "raises the switching cost.",
    ),
    (
        "PaymentMethod",
        "Migrate the customer to automatic payment (credit card or bank transfer) "
        "with a small bill-credit incentive.",
        "Electronic-check customers churn more; automated payment reduces friction "
        "and accidental lapses.",
    ),
    (
        "Total_Services",
        "Bundle additional services (streaming, backup, security) at a discount.",
        "Customers with fewer subscribed services have lower switching costs and "
        "churn more readily.",
    ),
    (
        "PaperlessBilling",
        "Send proactive bill summaries and cost-saving tips via email.",
        "Paperless-billing customers are more digitally engaged; timely "
        "communication reduces bill-shock churn.",
    ),
    (
        "SeniorCitizen",
        "Assign a dedicated senior-support concierge or simplified billing plan.",
        "Senior customers may need extra support; proactive outreach reduces churn "
        "driven by confusion or poor experience.",
    ),
    (
        "Avg_Monthly_Spend",
        "Review the billing history and offer a loyalty price-match.",
        "High average spend relative to tenure signals possible price sensitivity.",
    ),
    (
        "Is_High_Value",
        "Escalate to a dedicated account manager for a retention call.",
        "High-value customers represent disproportionate revenue; a personalised "
        "touch has the highest ROI.",
    ),
    (
        "Is_Long_Term",
        "Reward tenure with a loyalty gift (e.g. free month, device upgrade).",
        "Long-term customers in churn risk may feel undervalued; recognition "
        "reinforces loyalty.",
    ),
]


def business_recommendations(
    predict_output: Dict[str, Any],
    max_recommendations: int = 4,
) -> List[Dict[str, str]]:
    """Generate personalised retention recommendations from SHAP output.

    Recommendations are conditioned on the *actual* top SHAP features for
    this specific customer — not a static lookup keyed only on churn=Yes.
    A retained customer (prediction=0) receives no retention actions (no action
    needed), but at-risk customers receive targeted recommendations for their
    specific risk drivers.

    Parameters
    ----------
    predict_output : dict
        Output of ``predict_customer()``.
    max_recommendations : int
        Cap on number of recommendations returned.

    Returns
    -------
    list[dict]
        Each dict: ``{"recommendation": str, "rationale": str,
        "driving_feature": str, "shap_value": float}``
    """
    if predict_output["prediction"] == 0:
        return [
            {
                "recommendation": "No immediate retention action required.",
                "rationale": (
                    f"Model predicts retention (churn probability "
                    f"{predict_output['churn_probability']:.1%}, below threshold)."
                ),
                "driving_feature": "N/A",
                "shap_value": 0.0,
            }
        ]

    recommendations: List[Dict[str, str]] = []
    matched_rules: set = set()

    # Sort top features by absolute SHAP value (already sorted, but re-sort for safety)
    top_features = sorted(
        predict_output["top_features"], key=lambda x: abs(x["shap_value"]), reverse=True
    )

    for feat_dict in top_features:
        if len(recommendations) >= max_recommendations:
            break
        feature_name = feat_dict["feature"]
        shap_val = feat_dict["shap_value"]

        # Only act on features that push *toward* churn (positive SHAP for class 1)
        if shap_val <= 0:
            continue

        for substring, rec_text, rationale in _RECOMMENDATION_RULES:
            if substring.lower() in feature_name.lower() and substring not in matched_rules:
                recommendations.append(
                    {
                        "recommendation": rec_text,
                        "rationale": (
                            f"'{feature_name}' is a top churn driver for this customer "
                            f"(SHAP={shap_val:+.3f}). {rationale}"
                        ),
                        "driving_feature": feature_name,
                        "shap_value": round(shap_val, 4),
                    }
                )
                matched_rules.add(substring)
                break

    if not recommendations:
        recommendations.append(
            {
                "recommendation": "Review customer account and offer a general loyalty discount.",
                "rationale": (
                    "Model flags this customer as at risk but no specific "
                    "feature-level driver maps to a standard playbook action. "
                    "A personalised outreach call is recommended."
                ),
                "driving_feature": "multiple",
                "shap_value": 0.0,
            }
        )

    return recommendations
