# Design Decisions Log

This document records the engineering and modelling judgment calls made during the Customer Churn Prediction project.

## 1. Data Source & Quality

- **Synthetic Data Fallback**: The project specifies the Telco Customer Churn dataset from Kaggle. A `generate_data.py` script was created to bootstrap a schema-faithful synthetic substitute (7,043 rows, 21 columns, matching class imbalance) so the notebook can run autonomously even if the raw CSV is not manually downloaded.
- **`TotalCharges` Imputation**: In the original dataset, 11 rows have `tenure = 0` and a blank string for `TotalCharges`. Since these customers haven't been billed yet, imputing `0.0` is the most logically sound decision. Any other non-numeric `TotalCharges` values are dropped as genuine data errors.
- **Redundant Categorical Levels**: Levels like "No internet service" and "No phone service" were collapsed into "No" across relevant columns. The parent columns (`InternetService`, `PhoneService`) already encode this information, so collapsing these levels reduces cardinality and prevents spurious tree splits.

## 2. Feature Engineering

- **`Avg_Monthly_Spend`**: Defined as `TotalCharges / max(tenure, 1)`. This is often a cleaner signal than `TotalCharges` (which is highly collinear with tenure). The `max(tenure, 1)` guards against divide-by-zero errors.
- **`Tenure_Group` and `Charge_Category`**: Binned continuous variables. While tree models can find their own split points, explicitly binning these according to EDA insights helps linear models (like Logistic Regression) capture non-linear relationships.
- **`Total_Services_Subscribed`**: A count of all binary service add-ons. This serves as a proxy for "switching cost" — a known structural driver of churn.

## 3. Preprocessing & Leakage Prevention

- **Encoding Nominal vs Binary**: 
  - Two-level categorical features (e.g., `gender`, `Partner`) use `OneHotEncoder(drop='if_binary')` to produce a single 0/1 column.
  - Multi-level nominal features (e.g., `InternetService`, `PaymentMethod`) use `OneHotEncoder(drop='first')` to prevent multicollinearity. `LabelEncoder` was avoided as it imposes an artificial ordinal relationship.
- **Scaling Strategy**: `StandardScaler` is applied to all numeric features. While tree models are scale-invariant, linear models require scaled inputs. A single `ColumnTransformer` applying scaling universally keeps the pipeline simple and prevents duplicate logic in the serving layer.
- **Strict Separation**: The `ColumnTransformer` is fit *exclusively* on the training split. This is explicitly called out to prevent test-set distribution statistics from leaking into the model.

## 4. Class Imbalance Handling

- **Dual-Strategy Evaluation**: The dataset has a ~26% positive rate. Rather than assuming one method is superior, two configurations were compared:
  1. `class_weight='balanced'` inside the estimators.
  2. SMOTE (Synthetic Minority Over-sampling Technique) applied *only* to the training data.
- **SMOTE Boundary**: SMOTE is strictly applied post-split to the training data. Applying it beforehand is a common anti-pattern that inflates test-set performance artificially.

## 5. Model Evaluation & Selection

- **Primary Metric (`PR-AUC`)**: Accuracy is a misleading metric for imbalanced datasets (predicting "No Churn" universally yields ~74% accuracy). ROC-AUC can also be overly optimistic when true negatives are abundant. Precision-Recall AUC (PR-AUC) focuses explicitly on the positive (minority) class, making it the most defensible primary metric.
- **Threshold Tuning**: The default `0.5` classification threshold assumes equal cost for false positives and false negatives. In churn prediction, missing a churner (false negative) is typically far more expensive than offering an unnecessary retention discount (false positive). The threshold is tuned to maximise `F1`, pulling the decision boundary down to catch more true churners.

## 6. Model Explainability & Operationalisation

- **SHAP vs Impurity Importance**: `sklearn`'s native feature importance (Gini/impurity) is heavily biased toward continuous/high-cardinality features. Permutation importance (on the test set) and SHAP values provide a more trustworthy explanation of generalisable drivers.
- **Personalised Recommendations**: The `predict_customer()` function does not just output a probability. It uses local SHAP values to identify *why* a specific customer is at risk, and maps those specific features to actionable business recommendations, rather than relying on a static lookup table.
