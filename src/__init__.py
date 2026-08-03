"""
src — Customer Churn Prediction helper modules.

Submodules
----------
preprocessing : data loading, cleaning, feature engineering, and preprocessor construction
modeling      : model factory, training/evaluation helpers, hyperparameter search, threshold analysis
explain       : SHAP computation, predict_customer(), and business recommendations
"""
from src.preprocessing import load_and_clean, engineer_features, build_preprocessor
from src.modeling import (
    get_models,
    train_evaluate,
    hyperparameter_search,
    plot_roc_curves,
    threshold_analysis,
)
from src.explain import compute_shap, predict_customer, business_recommendations

__all__ = [
    "load_and_clean",
    "engineer_features",
    "build_preprocessor",
    "get_models",
    "train_evaluate",
    "hyperparameter_search",
    "plot_roc_curves",
    "threshold_analysis",
    "compute_shap",
    "predict_customer",
    "business_recommendations",
]
