# Customer Churn Prediction

> A production-quality ML portfolio project for Telco customer churn prediction.

---

## 📋 Dataset

**Source**: Telco Customer Churn (IBM/Kaggle schema — synthetic schema-faithful substitute generated automatically if the real CSV is absent)

- **Rows**: 7,043 customers
- **Features**: 21 (demographics, service subscriptions, contract details, billing)
- **Target**: `Churn` (binary: Yes/No → 1/0)
- **Class balance**: ~26.5% churners / ~73.5% retained

---

## 🚀 Setup & Reproduction

### Requirements

```bash
pip install -r requirements.txt
```

### Quick Start

```bash
# 1. Generate synthetic data (or place the real Kaggle CSV at data/telco_churn.csv)
python generate_data.py

# 2. Open and run the portfolio notebook
jupyter notebook churn_prediction.ipynb

# 3. (Optional) Run notebook non-interactively end-to-end
jupyter nbconvert --to notebook --execute --inplace churn_prediction.ipynb
```

### Project Structure

```
ML Project ITR/
├── data/
│   └── telco_churn.csv          # Raw dataset (synthetic or real Kaggle CSV)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # load_and_clean, engineer_features, build_preprocessor
│   ├── modeling.py               # get_models, train_evaluate, hyperparameter_search, plots
│   └── explain.py                # compute_shap, predict_customer, business_recommendations
├── models/
│   ├── best_model.joblib         # Saved best-performing model
│   ├── preprocessor.joblib       # Saved fitted ColumnTransformer
│   └── model_comparison.json     # All model metrics (JSON)
├── churn_prediction.ipynb        # ← Main portfolio deliverable (20 sections)
├── generate_data.py              # Synthetic data generator
├── requirements.txt
├── README.md
└── DECISIONS.md                  # All judgment calls documented
```

---

## 🏆 Best Model & Score

| Metric | Value |
|---|---|
| **Model** | See Section 12 output (top PR-AUC after tuning) |
| **Primary Metric** | **PR-AUC** (not accuracy — imbalanced dataset) |
| **Classification Threshold** | Tuned via F1-maximisation (Section 14) |
| **Imbalance Strategy** | `class_weight='balanced'` vs SMOTE (compared) |

> **Why PR-AUC?** At a 74/26 class balance, a model predicting "No Churn" always achieves 74% accuracy. PR-AUC focuses on the minority class (churners) and is the correct quality gate for this problem.

---

## 💡 Key Findings

### Top Churn Predictors (SHAP-verified)

1. **Contract = Month-to-month** — by far the strongest predictor. Monthly customers churn 3–4× more than two-year subscribers.
2. **Tenure (short)** — the first 12 months are the critical loyalty window.
3. **Monthly Charges (high)** — high-bill customers perceive poor value and leave.
4. **Internet Service = Fiber optic** — premium segment with high churn, likely a value-perception gap.
5. **Tech Support = No / Online Security = No** — missing add-ons signal disengagement.
6. **Total Services Subscribed (low)** — fewer services = lower switching cost.
7. **Payment Method = Electronic check** — proxy for lower commitment/automation.

### Business Insights

- **Contract upgrades** are the highest-ROI retention lever.
- **Onboarding programmes** targeting the first 12 months have outsized impact.
- **Bundle add-ons** at a discount to increase switching cost for high-churn-risk segments.
- **Auto-pay enrolment** can reduce accidental churn from electronic-check users.

---

## 🔬 Methodology

| Decision | Choice | Rationale |
|---|---|---|
| Primary metric | PR-AUC | Most informative at 74/26 imbalance |
| Imbalance | `class_weight` + SMOTE (compared) | Neither alone is always best |
| Encoding | OHE for nominals, binary for Yes/No | No ordinal relationship in nominals |
| Threshold | Tuned (F1 maximisation) | Default 0.5 under-predicts churners |
| Hyperparameter search | RandomizedSearchCV ≤30 iter, scored on F1 | Bounded cost, correct metric |
| Leakage prevention | Preprocessor fit on train only | Stated explicitly in notebook |

---

## 📁 Notebook Sections

| # | Section | Key Output |
|---|---|---|
| 1 | Introduction | Problem framing, industry context |
| 2 | Setup | Libraries, RANDOM_STATE=42 constant |
| 3 | Load & Inspect | Raw data exploration |
| 4 | Data Cleaning | TotalCharges fix, dedup, dtype fixes |
| 5 | EDA | 8+ plots with written business insights |
| 6 | Feature Engineering | 6 engineered features + validation |
| 7 | Encoding | OHE strategy + leakage prevention |
| 8 | Feature Scaling | StandardScaler rationale |
| 9 | Train-Test Split | Stratified 80/20, balance verification |
| 10 | Class Imbalance | SMOTE + class_weight comparison |
| 11 | Model Training | 6 models × 2 configs = 12 trained |
| 12 | Model Evaluation | PR-AUC table, ROC/PR curves |
| 13 | Hyperparameter Tuning | RF, XGB, CatBoost tuned |
| 14 | Threshold Tuning | Precision/recall/F1 vs. threshold |
| 15 | Feature Importance | Native + permutation importance |
| 16 | SHAP Explainability | Summary + waterfall (churner + retainer) |
| 17 | predict_customer() | End-to-end customer scoring |
| 18 | Business Recommendations | SHAP-conditioned actions |
| 19 | Save Artifacts | joblib + JSON |
| 20 | Conclusion | Summary, findings, future work |

---

## ⚙️ Configuration

All randomness is seeded via `RANDOM_STATE = 42` (single constant defined in the notebook, never re-typed as a literal).

---

## 📝 License

MIT — free to use and adapt for portfolio or commercial purposes.
