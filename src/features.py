# ---------------------------------------------------------
# src/features.py - Centralized Business Logic
# ---------------------------------------------------------
import pandas as pd
import numpy as np

def calculate_ltv(loan_amount, property_value):
    """
    Calculates Loan-to-Value Ratio.
    Safe against DivisionByZero.
    """
    if property_value is None or property_value == 0:
        return 0.0
    return (loan_amount / property_value) * 100

def calculate_lti(loan_amount, income):
    """
    Calculates Loan-to-Income Ratio.
    Handles 0 income by returning a high ratio cap.
    """
    if income is None or income <= 0:
        return 999.0 # Max Risk
    return loan_amount / income

def calculate_disposable_income(income, loan_amount, interest_rate=0.005):
    """
    Estimates monthly disposable income after mortgage payment.
    Default interest rate proxy is 0.5% monthly (~6% annual).
    """
    est_payment = loan_amount * interest_rate
    return income - est_payment

def calculate_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates the 'Alpha Signals' (Interaction Flags).
    Expects a DataFrame with 'LTV' and 'Credit_Score'.
    """
    df = df.copy()
    
    # 1. Risky LTV (>80%)
    df['risky_ltv'] = (df['LTV'] > 80).astype(int)
    
    # 2. Subprime Credit (<650)
    df['risky_credit'] = (df['Credit_Score'] < 650).astype(int)
    
    # 3. Double Jeopardy (Both)
    df['double_risk'] = df['risky_ltv'] * df['risky_credit']
    
    return df