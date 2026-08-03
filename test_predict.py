import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib

from src.preprocessing import get_feature_names
from src.explain import predict_customer

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessor.joblib")
preprocessor = joblib.load(PREPROCESSOR_PATH)
feature_names = get_feature_names(preprocessor)

models_to_test = {
    "Random Forest": "models/rf_model.joblib",
    "Decision Tree": "models/dt_model.joblib",
    "Naive Bayes": "models/nb_model.joblib",
    "Logistic Regression": "models/lr_model.joblib",
}

test_customer = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 89.50, "TotalCharges": 268.50,
}

for name, path in models_to_test.items():
    try:
        print(f"\n--- Testing {name} ---")
        model = joblib.load(os.path.join(PROJECT_ROOT, path))
        
        import shap
        if "Logistic" in name:
            explainer = shap.LinearExplainer(model, shap.maskers.Independent(np.zeros((1, len(feature_names))))) 
        elif "Naive" in name:
            explainer = shap.KernelExplainer(model.predict_proba, np.zeros((1, len(feature_names))))
        else:
            explainer = shap.TreeExplainer(model)
            
        pred = predict_customer(test_customer, preprocessor, model, 0.5, explainer, feature_names)
        print(f"SUCCESS! Prediction: {pred['prediction']}, Prob: {pred['churn_probability']:.3f}")
    except Exception as e:
        import traceback
        print(f"FAILED! Error: {type(e).__name__}: {e}")
        traceback.print_exc()
