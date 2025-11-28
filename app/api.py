# ---------------------------------------------------------
# app/api.py - The "Hybrid" Risk Engine (AI + Policy Rules)
# ---------------------------------------------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
import os
import traceback 

# Initialize FastAPI
app = FastAPI(title="AuraFin Risk Engine", version="2.0.0")

# ---------------------------------------------------------
# 1. Load Artifacts
# ---------------------------------------------------------
MODEL_PATH = "models/xgb_champion.pkl"
ENCODER_PATH = "models/label_encoders.pkl"
CONFIG_PATH = "models/threshold_config.json"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(" Model file not found. Run Notebook 04 first.")

print(" Loading AuraFin Engine...")
try:
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
        OPTIMAL_THRESHOLD = config.get("optimal_threshold", 0.50)
    print(f" System Online. Threshold set to: {OPTIMAL_THRESHOLD:.4f}")
except Exception as e:
    print(f" Error loading artifacts: {e}")
    raise e

# ---------------------------------------------------------
# 2. Define Input Schema
# ---------------------------------------------------------
class LoanApplication(BaseModel):
    # Numerical Fields
    loan_amount: float
    income: float
    property_value: float
    Credit_Score: int
    LTV: float
    term: float = 360.0
    dtir1: float = 40.0
    age: str = "35-44"

    # Categorical Fields
    Gender: str
    Region: str
    loan_type: str
    loan_purpose: str
    credit_type: str
    approv_in_adv: str = "nopre"
    Credit_Worthiness: str = "l1"
    open_credit: str = "nopc"
    business_or_commercial: str = "nob/c"
    Neg_ammortization: str = "not_neg"
    interest_only: str = "not_int"
    lump_sum_payment: str = "not_lpsm"
    occupancy_type: str = "pr"
    total_units: str = "1U"
    co_applicant_credit_type: str = "CIB"
    submission_of_application: str = "to_inst"
    loan_limit: str = "cf"

# ---------------------------------------------------------
# 3. THE POLICY ENGINE (Hard Rules / Knockouts)
# ---------------------------------------------------------
def check_policy_rules(data: LoanApplication):
    """
    Applies 'Common Sense' banking rules before the AI Model.
    Returns: (is_rejected, reason)
    """
    # RULE 1: Minimum Income Check
    # Even if FICO is 850, you can't pay a loan with $0 income.
    if data.income <= 0:
        return True, "POLICY REJECT: Income cannot be zero or negative."
    
    # RULE 2: Minimum Credit Score (Subprime Floor)
    # Banks often have a hard cutoff (e.g., 500) below which they won't even look.
    if data.Credit_Score < 500:
        return True, f"POLICY REJECT: Credit Score {data.Credit_Score} is below minimum eligibility (500)."
    
    # RULE 3: Minimum Property Value
    # We don't mortgage shacks worth $10k.
    if data.property_value < 50000:
        return True, "POLICY REJECT: Property value is below collateral minimum ($50k)."
        
    # RULE 4: Extreme LTV Check
    # LTV > 105% (Under water) is almost always illegal or auto-reject
    calculated_ltv = (data.loan_amount / data.property_value) * 100 if data.property_value > 0 else 999
    if calculated_ltv > 105:
        return True, f"POLICY REJECT: LTV {calculated_ltv:.1f}% exceeds maximum regulatory limit (105%)."

    # If all pass
    return False, "Passed"

# ---------------------------------------------------------
# 4. Preprocessing Logic (The Safety Net)
# ---------------------------------------------------------
def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # A. Feature Engineering
    income_val = df['income'].fillna(0).replace(0, 1.0)
    df['LTI'] = df['loan_amount'] / income_val

    if 'LTV' not in df.columns or df['LTV'].iloc[0] == 0:
        prop_safe = df['property_value'].fillna(0).replace(0, 1.0)
        df['LTV'] = (df['loan_amount'] / prop_safe) * 100

    df['est_monthly_payment'] = df['loan_amount'] * 0.005
    df['disposable_income'] = df['income'] - df['est_monthly_payment']

    df['risky_ltv'] = (df['LTV'] > 80).astype(int)
    df['risky_credit'] = (df['Credit_Score'] < 650).astype(int)
    df['double_risk'] = df['risky_ltv'] * df['risky_credit']

    # B. Label Encoding
    for col, le in encoders.items():
        if col in df.columns:
            val = str(df.loc[0, col])
            if val in le.classes_:
                df.loc[0, col] = le.transform([val])[0]
            else:
                df.loc[0, col] = le.transform([le.classes_[0]])[0]

    # C. Column Alignment
    try:
        if hasattr(model, "feature_names_in_"):
            expected_cols = model.feature_names_in_
        else:
            expected_cols = model.get_booster().feature_names
    except:
        print(" Warning: Could not detect feature names from model.")
        return df

    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0
    
    return df[expected_cols]

# ---------------------------------------------------------
# 5. Prediction Endpoint
# ---------------------------------------------------------
@app.post("/predict_risk", tags=["Risk Engine"])
def predict_risk(application: LoanApplication):
    try:
        # --- STEP 1: RUN POLICY ENGINE (The "Guardrails") ---
        # This fixes the logical fallacies (0 income, etc.)
        is_knockout, rejection_reason = check_policy_rules(application)
        
        if is_knockout:
            return {
                "decision": "REJECT",
                "risk_score": 1.00, # Max Risk
                "threshold_used": OPTIMAL_THRESHOLD,
                "financial_analysis": {
                    "risk_weighted_exposure": 0 # No exposure because we rejected
                },
                "policy_flag": rejection_reason # Tell the UI why
            }

        # --- STEP 2: RUN AI MODEL (If Policy Passed) ---
        raw_data = application.dict()
        processed_df = preprocess_input(raw_data)
        processed_df = processed_df.apply(pd.to_numeric, errors='coerce').fillna(0)

        probs = model.predict_proba(processed_df)
        risk_probability = float(probs[0][1])

        decision = "REJECT" if risk_probability >= OPTIMAL_THRESHOLD else "APPROVE"
        
        # Calculate Exposure
        exposure = application.loan_amount * 0.60 * risk_probability

        return {
            "decision": decision,
            "risk_score": risk_probability,
            "threshold_used": OPTIMAL_THRESHOLD,
            "financial_analysis": {
                "risk_weighted_exposure": exposure
            }
        }

    except Exception as e:
        print(" API CRASHED:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
# ---------------------------------------------------------
# 6. Health Check Endpoint
# ---------------------------------------------------------
@app.get("/", tags=["Health"])
def health_check():
    return {"status": "online", "engine": "AuraFin Risk Engine v2.0"}