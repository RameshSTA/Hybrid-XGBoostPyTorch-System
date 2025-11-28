# ---------------------------------------------------------
#  Unit Tests for Business Logic
# ---------------------------------------------------------
import pytest
import pandas as pd
from src.features import calculate_ltv, calculate_lti, calculate_risk_flags

def test_ltv_calculation():
    # Scenario: Standard Case
    assert calculate_ltv(80000, 100000) == 80.0
    
    # Scenario: Zero Property Value (Should not crash)
    assert calculate_ltv(50000, 0) == 0.0

def test_lti_calculation():
    # Scenario: Standard Case
    assert calculate_lti(100000, 50000) == 2.0
    
    # Scenario: Zero Income (Should return High Risk Cap)
    assert calculate_lti(100000, 0) == 999.0

def test_risk_flags():
    # Create dummy data
    data = pd.DataFrame({
        'LTV': [50, 90, 90],
        'Credit_Score': [700, 700, 600] 
    })
    
    result = calculate_risk_flags(data)
    
    # Case 1: Safe (50 LTV, 700 FICO) -> Should be 0
    assert result.loc[0, 'double_risk'] == 0
    
    # Case 2: High LTV only (90 LTV, 700 FICO) -> Should be 0
    assert result.loc[1, 'double_risk'] == 0
    
    # Case 3: Double Risk (90 LTV, 600 FICO) -> Should be 1
    assert result.loc[2, 'double_risk'] == 1