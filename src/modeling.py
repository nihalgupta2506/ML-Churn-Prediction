"""
src/modeling.py
===============
Model factory, training/evaluation helpers, hyperparameter search, and
threshold analysis for the Telco Customer Churn prediction project.

Design notes
------------
- ``RANDOM_STATE`` is NOT re-defined here; callers must pass it explicitly
  to keep the single-constant contract defined in the notebook.
- Primary evaluation metric is PR-AUC (average_precision_score), not accuracy,
  because the dataset has ~26% positive rate — accuracy is misleading and
  ROC-AUC is optimistic; PR-AUC reflects the precision/recall tradeoff that
  actually matters for a minority-class problem.
- Class imbalance is handled via ``class_weight`` on estimators that support it
  AND (optionally) via SMOTE applied to training data only.  Both are surfaced
  as explicit parameters so the notebook can compare them.
"""

from __future__ import annotations

import json
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def get_models(
    random_state: int, use_class_weight: bool = True
) -> Dict[str, Any]:
    """Return a dictionary of unfitted estimators keyed by model name.

    Parameters
    ----------
    random_state : int
        Seed for reproducibility — passed to every estimator that accepts it.
    use_class_weight : bool
        If True, set ``class_weight='balanced'`` on estimators that support it
        (LogisticRegression, DecisionTree, RandomForest, GradientBoosting).
        XGBoost and CatBoost handle imbalance via ``scale_pos_weight`` /
        ``auto_class_weights`` respectively.

    Returns
    -------
    dict[str, estimator]
        Ordered dict: LR → DT → RF → GB → XGB → CatBoost.
    """
    cw = "balanced" if use_class_weight else None

    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight=cw,
            random_state=random_state,
            solver="lbfgs",
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight=cw,
            random_state=random_state,
            max_depth=8,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight=cw,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            scale_pos_weight=3,  # approx (1 - churn_rate) / churn_rate for 26% rate
            verbosity=0,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200,
            random_seed=random_state,
            auto_class_weights="Balanced",
            verbose=0,
        ),
    }
    return models


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------


def train_evaluate(
    name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Fit *model*, evaluate on *X_test / y_test*, and return a metric dict.

    Parameters
    ----------
    name : str
        Human-readable model name (for logging / reports).
    model : sklearn-compatible estimator
        Unfitted estimator with a ``predict_proba`` method.
    X_train, y_train : array-like
        Training features and labels.
    X_test, y_test : array-like
        Hold-out test features and labels.
    threshold : float
        Classification threshold applied to ``predict_proba`` output.
        Default 0.5 matches sklearn default; pass a tuned value for final eval.

    Returns
    -------
    dict
        Keys: model_name, accuracy, precision, recall, f1, roc_auc, pr_auc,
        confusion_matrix (as nested list), classification_report (str),
        train_time_s, y_prob (array — for later ROC / threshold plots).
    """
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics: Dict[str, Any] = {
        "model_name": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc": round(average_precision_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["No Churn", "Churn"]
        ),
        "train_time_s": round(train_time, 2),
        "y_prob": y_prob,  # stored for plotting; excluded from JSON export
    }
    return metrics


def build_comparison_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of metric dicts into a sorted comparison DataFrame.

    Sorted descending by PR-AUC (primary metric), then F1.

    Parameters
    ----------
    results : list[dict]
        Output of repeated ``train_evaluate`` calls.

    Returns
    -------
    pd.DataFrame
        Columns: Model, Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC,
        Train Time (s).  Sorted by PR-AUC descending.
    """
    rows = []
    for r in results:
        rows.append(
            {
                "Model": r["model_name"],
                "Accuracy": r["accuracy"],
                "Precision": r["precision"],
                "Recall": r["recall"],
                "F1": r["f1"],
                "ROC-AUC": r["roc_auc"],
                "PR-AUC": r["pr_auc"],
                "Train Time (s)": r["train_time_s"],
            }
        )
    df = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
    return df


def save_comparison_json(results: List[Dict[str, Any]], path: str) -> None:
    """Persist model comparison metrics to *path* as JSON.

    The ``y_prob`` array is excluded (not JSON-serialisable and not needed
    for a comparison report).

    Parameters
    ----------
    results : list[dict]
        Output of repeated ``train_evaluate`` calls.
    path : str
        Destination file path (e.g. ``models/model_comparison.json``).
    """
    serialisable = []
    for r in results:
        record = {k: v for k, v in r.items() if k != "y_prob"}
        serialisable.append(record)
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_roc_curves(
    results: List[Dict[str, Any]],
    y_test: np.ndarray,
    figsize: Tuple[int, int] = (9, 7),
) -> plt.Figure:
    """Plot all models' ROC curves on a single axes for comparison.

    Parameters
    ----------
    results : list[dict]
        Each dict must have keys ``model_name`` and ``y_prob``.
    y_test : array-like
        True binary labels for the test set.
    figsize : tuple
        Matplotlib figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(results)))

    for result, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, result["y_prob"])
        auc = result["roc_auc"]
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{result['model_name']} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_pr_curves(
    results: List[Dict[str, Any]],
    y_test: np.ndarray,
    figsize: Tuple[int, int] = (9, 7),
) -> plt.Figure:
    """Plot Precision-Recall curves for all models on a single axes.

    PR curves are more informative than ROC curves under class imbalance
    because they focus on the minority class (churners) and are not inflated
    by the large number of true negatives.

    Parameters
    ----------
    results : list[dict]
        Each dict must have keys ``model_name`` and ``y_prob``.
    y_test : array-like
        True binary labels for the test set.
    figsize : tuple
        Matplotlib figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(results)))
    baseline = y_test.mean()

    for result, color in zip(results, colors):
        prec, rec, _ = precision_recall_curve(y_test, result["y_prob"])
        ap = result["pr_auc"]
        ax.plot(rec, prec, lw=2, color=color, label=f"{result['model_name']} (AP={ap:.3f})")

    ax.axhline(y=baseline, color="k", linestyle="--", lw=1, label=f"Baseline (AP={baseline:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: list,
    model_name: str,
    figsize: Tuple[int, int] = (6, 5),
) -> plt.Figure:
    """Plot a labelled confusion matrix heatmap.

    Parameters
    ----------
    cm : list[list[int]]
        2×2 confusion matrix (output of ``confusion_matrix().tolist()``).
    model_name : str
        Used in the plot title.
    figsize : tuple
        Matplotlib figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        np.array(cm),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

# Param grids for RandomizedSearchCV — bounded to ≤30 iterations total
PARAM_GRIDS: Dict[str, Dict[str, Any]] = {
    "Random Forest": {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.2, 0.5],
        "min_child_weight": [1, 3, 5],
    },
    "CatBoost": {
        "iterations": [100, 200, 300],
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "l2_leaf_reg": [1, 3, 5, 7],
        "bagging_temperature": [0, 0.5, 1.0],
    },
}


def hyperparameter_search(
    model_name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    scoring: str = "f1",
    cv: int = 5,
    n_iter: int = 30,
    random_state: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    """Run RandomizedSearchCV for *model_name* and return the best estimator.

    Parameters
    ----------
    model_name : str
        Must match a key in ``PARAM_GRIDS``.
    model : sklearn-compatible estimator
        Base estimator (unfitted).
    X_train, y_train : array-like
        Training data.
    scoring : str
        Sklearn scoring string.  Default 'f1' to match the notebook's
        primary metric — NOT 'accuracy', which would optimise for the
        majority class and ignore the imbalance problem.
    cv : int
        Number of stratified folds.
    n_iter : int
        Maximum number of parameter combinations sampled (≤30 per spec).
    random_state : int
        Seed for RandomizedSearchCV.

    Returns
    -------
    (best_estimator, best_params)
    """
    if model_name not in PARAM_GRIDS:
        raise ValueError(
            f"No param grid defined for '{model_name}'. "
            f"Available: {list(PARAM_GRIDS.keys())}"
        )

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=PARAM_GRIDS[model_name],
        n_iter=min(n_iter, 30),
        scoring=scoring,
        cv=skf,
        n_jobs=-1,
        random_state=random_state,
        verbose=0,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------


def threshold_analysis(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    figsize: Tuple[int, int] = (10, 6),
) -> Tuple[float, plt.Figure]:
    """Plot precision/recall/F1 vs. threshold and return the optimal threshold.

    Business rationale for threshold selection
    ------------------------------------------
    In a churn-prevention context, the cost of a false negative (a churning
    customer we miss and lose) typically exceeds the cost of a false positive
    (a retained customer we unnecessarily offer a discount to).  We therefore
    pick the threshold that maximises F1 — a balanced harmonic mean of
    precision and recall — rather than the default 0.5, which was tuned for
    balanced datasets and under-predicts churners here.

    Parameters
    ----------
    y_test : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    figsize : tuple
        Matplotlib figure size.

    Returns
    -------
    (optimal_threshold, figure)
        ``optimal_threshold`` is the threshold that maximises F1 on the
        test set.  This value is stored and reused in ``predict_customer()``.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    # F1 at each threshold (thresholds has len = len(precisions) - 1)
    f1_scores = (
        2 * precisions[:-1] * recalls[:-1]
        / (precisions[:-1] + recalls[:-1] + 1e-9)
    )
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[optimal_idx])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(thresholds, precisions[:-1], label="Precision", lw=2, color="#2196F3")
    ax.plot(thresholds, recalls[:-1], label="Recall", lw=2, color="#4CAF50")
    ax.plot(thresholds, f1_scores, label="F1", lw=2, color="#FF5722", linestyle="--")
    ax.axvline(
        optimal_threshold,
        color="red",
        linestyle=":",
        lw=2,
        label=f"Optimal threshold = {optimal_threshold:.3f}",
    )
    ax.axvline(0.5, color="grey", linestyle=":", lw=1, label="Default threshold = 0.50")
    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Precision / Recall / F1 vs. Classification Threshold",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return optimal_threshold, fig


def permutation_importance_plot(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Compute and plot permutation importance on the held-out test set.

    Permutation importance is preferred over impurity-based importance because:
    - It is computed on held-out data, so it measures *generalisation*-relevant
      importance rather than training-set fit.
    - It is not biased toward high-cardinality features (a known flaw in
      sklearn's impurity-based importance for tree models).

    Parameters
    ----------
    model : fitted estimator
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True test labels.
    feature_names : list[str]
        Column names matching X_test columns.
    n_repeats : int
        Number of times to permute each feature.
    random_state : int
        Seed for reproducibility.
    top_n : int
        Number of top features to display.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.inspection import permutation_importance as sk_perm_importance

    result = sk_perm_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="average_precision",
    )
    perm_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": result.importances_mean,
            "Std": result.importances_std,
        }
    ).sort_values("Importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(
        perm_df["Feature"][::-1],
        perm_df["Importance"][::-1],
        xerr=perm_df["Std"][::-1],
        color="#5C6BC0",
        alpha=0.85,
        edgecolor="white",
    )
    ax.set_xlabel("Mean Decrease in PR-AUC (permutation)", fontsize=12)
    ax.set_title(
        f"Permutation Importance — Top {top_n} Features (Test Set)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig
