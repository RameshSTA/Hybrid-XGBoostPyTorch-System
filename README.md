# AuraFin: Hybrid XGBoost/PyTorch System with Finacial Threshold Optimization

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-009688?style=for-the-badge&logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-Champion-orange?style=for-the-badge&logo=xgboost)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)
![Impact](https://img.shields.io/badge/Net%20Savings-%2421.8M-success?style=for-the-badge)

> **"A Production-Grade Risk Engine that replaces 'Statistical Accuracy' with 'Net Profit Optimization'."**

---

## Executive Summary

In credit risk modeling, standard Machine Learning algorithms optimize for **Accuracy**. On highly imbalanced datasets (e.g., 94% good loans), a model can achieve **94% accuracy** by simply predicting "No Default" for everyone.

**The Business Problem:**
This "Accuracy Trap" is catastrophic for banks because the cost of errors is asymmetric:
* **False Positive (Rejection):** Cost = **$6,000** (Lost Interest Profit).
* **False Negative (Default):** Cost = **$60,000** (Lost Principal).

**The AuraFin Solution:**
We built a **Cost-Sensitive Engine** that explicitly minimizes the global financial loss function. By penalizing missed defaults 15x more than false alarms and optimizing the decision threshold, we achieved:

| Metric | Standard "Vanilla" Model | **AuraFin Risk Engine** |
| :--- | :--- | :--- |
| **Optimization Goal** | Maximize Accuracy | **Minimize Financial Loss** |
| **Recall (Defaults)** | ~0% (Misses Risky Loans) | **71%** (Catches Risky Loans) |
| **Projected Loss** | $35.3 Million | **$13.5 Million** |
| **NET CAPITAL SAVED** | - | **$21.8 Million per Batch** |

---

## Engineering Methodology

This project follows a rigorous Data Science lifecycle, moving from Forensics to Engineering to Optimization.

### 1. Data Forensics (The Audit)
Before modeling, we performed a deep audit of the data integrity.
* **Leakage Detection:** We identified that `rate_of_interest` was missing for 99% of rejected loans. This confirmed it was a post-approval artifact. We removed it to prevent Data Leakage.
* **Distribution Analysis:** We validated that Income follows a **Log-Normal Distribution** (skewed by high earners). We used **Median Imputation** to prevent outlier bias.

### 2. Strategic Feature Engineering
We applied banking domain knowledge to engineer "Alpha Signals" that linearize risk for tree-based models:
* **LTV (Loan-to-Value):** Verified the "Skin in the Game" theory. Risk spikes significantly when LTV > 80%.
* **LTI (Loan-to-Income):** Engineered a ratio to measure repayment capacity ($Loan / Income$).
* **Interaction Flags:** Created specific boolean triggers for `Double_Risk` (High LTV + Subprime Credit).

### 3. Financial Threshold Optimization (The Math)
Most models default to a probability threshold of `0.50`. We rejected this approach. Instead, we solved for the optimal threshold $t$ by minimizing the **Global Cost Function**:

$$J(t) = (FN \times \$60,000) + (FP \times \$6,000)$$

* **The Anomaly:** The optimal threshold was found at **0.59** (higher than standard).
* **The Reason:** Because we used aggressive **Class Weights (15:1)** to handle the imbalance, the model became "paranoid," inflating risk scores. The optimization loop found that raising the bar to 0.59 recovered **$300k** in safe revenue without increasing default risk.

---

##  System Architecture

This project is deployed as a modern **Microservices Architecture**:

###  The Brain: FastAPI Backend (`app/api.py`)
A robust REST API that serves the model and enforces **Hard Policy Rules** (Guardrails).
* **Hybrid Logic:** Combines the XGBoost Model with a Rule Engine (e.g., auto-reject if Income is $0 or FICO < 500).
* **Validation:** Uses `Pydantic` for strict data validation.
* **Safety:** Handles edge cases (NaNs, Zero Division) via a robust preprocessing pipeline.

###  The Face: Streamlit Dashboard (`app/dashboard.py`)
An interactive "FinTech" interface for Loan Officers and Risk Executives.
* **Live Decisioning:** Real-time scoring with visual gauges.
* **Explainability:** Explains *why* a decision was made (e.g., "High Leverage").
* **Impact Analysis:** Visualizes the $21.8M savings curve.

---

##  Project Structure

A professional, modular structure ensuring reproducibility and scalability.

```text
AuraFin-Risk-Engine/
│
├── app/                        # Production Code (Microservices)
│   ├── api.py                  # FastAPI Backend (Logic & Policy Engine)
│   └── dashboard.py            # Streamlit Frontend (UI)
│
├── src/                        # Centralized Business Logic
│   └── features.py             # Feature Engineering Logic (DRY Principle)
│
├── notebooks/                  # The Data Science Laboratory
│   ├── 01_Forensics.ipynb      # Leakage & Imbalance Analysis
│   ├── 02_Preprocessing.ipynb  # Imputation & Cleaning
│   ├── 03_FeatureEng.ipynb     # LTV/LTI Engineering
│   ├── 04_Training.ipynb       # XGBoost vs PyTorch Benchmarking
│   └── 05_Optimization.ipynb   # Financial Threshold Tuning ($21M Savings)
│
├── models/                     # Serialized Artifacts
│   ├── xgb_champion.pkl        # Trained Model Artifact
│   ├── label_encoders.pkl      # Categorical Encoders
│   └── threshold_config.json   # Optimization Parameters
│
├── tests/                      # Quality Assurance
│   ├── test_api.py             # Integration Tests
│   └── test_features.py        # Unit Tests
│
├── images/                     # Visual Evidence Charts
├── Dockerfile                  # Containerization Config
├── requirements.txt            # Dependency Management
└── README.md                   # Project Documentation