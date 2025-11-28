# ---------------------------------------------------------
# tests/test_api.py - Integration Test for the API
# ---------------------------------------------------------
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_health_check():
    """Does the API wake up?"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_prediction_flow():
    """Does the Risk Engine return a valid decision?"""
    payload = {
        "loan_amount": 350000,
        "income": 9000,
        "property_value": 450000,
        "Credit_Score": 720,
        "LTV": 77.7,
        "Region": "North",
        "loan_purpose": "p1",
        "Gender": "Male",
        "loan_type": "type1",
        "credit_type": "CIB",
        "age": "35-44",
        "dtir1": 40.0
    }
    
    response = client.post("/predict_risk", json=payload)
    
    # Check if successful
    assert response.status_code == 200
    
    # Check response structure
    data = response.json()
    assert "decision" in data
    assert "risk_score" in data
    assert "threshold_used" in data
    
    # Sanity Check: Risk score should be a probability (0-1)
    assert 0 <= data["risk_score"] <= 1