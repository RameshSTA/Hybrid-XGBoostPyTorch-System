# ---------------------------------------------------------
# 
# ---------------------------------------------------------
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import os

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="AuraFin | AuraFin: Hybrid XGBoost/PyTorch System with Finacial Threshold Optimization",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. MODERN CSS (Professional, Clean, & Robust)
st.markdown("""
    <style>
    /* Global Settings */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1E293B;
        background-color: #F8FAFC; /* Slate 50 */
    }
    
    /* Card Styling */
    div.stExpander, div.css-1r6slb0 {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        padding: 20px;
    }
    
    /* Metrics Styling */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 2px solid #E2E8F0;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    label[data-testid="stMetricLabel"] {
        color: #64748B;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] {
        color: #0F172A;
        font-size: 28px;
        font-weight: 700;
    }
    
    /* Custom Badge Styling */
    .verdict-box {
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid;
        height: 200px; /* Fixed height for alignment */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .verdict-approve {
        background-color: #ECFDF5; /* Emerald 50 */
        border-color: #10B981;
        color: #047857;
    }
    .verdict-reject {
        background-color: #FEF2F2; /* Rose 50 */
        border-color: #F43F5E;
        color: #BE123C;
    }
    .verdict-title {
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
        opacity: 0.8;
    }
    .verdict-value {
        font-size: 36px;
        font-weight: 800;
        margin: 0;
    }
    
    /* Metric Box Styling (Similar to Verdict) */
    .metric-box {
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #E2E8F0;
        background-color: #FFFFFF;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-title {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
        color: #64748B;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #0F172A;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 2px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 14px;
        font-weight: 600;
        color: #64748B;
        padding-bottom: 12px;
    }
    .stTabs [aria-selected="true"] {
        color: #2563EB;
        border-bottom-color: #2563EB;
    }
    
    /* Gauge Card */
    .gauge-card {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        background-color: #FFFFFF;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        padding: 10px;
        height: 200px; /* Fixed height for alignment */
    }
    </style>
    """, unsafe_allow_html=True)

# API URL
API_URL = "http://127.0.0.1:8000/predict_risk"

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=50)
    st.markdown("### AuraFin Control")
    st.caption("Production v1.0 | XGBoost")
    st.markdown("---")
    
    st.markdown("** Financial Profile**")
    income = st.number_input("Monthly Income ($)", value=9500, step=500)
    loan_amt = st.number_input("Loan Amount ($)", value=320000, step=1000)
    prop_val = st.number_input("Property Value ($)", value=450000, step=1000)
    
    ltv = (loan_amt / prop_val) * 100 if prop_val > 0 else 0
    st.markdown(f"**LTV Ratio:** `{ltv:.1f}%`")
    if ltv > 80:
        st.warning("High Leverage Risk")
    
    st.markdown("---")
    st.markdown("** Applicant Details**")
    credit_score = st.slider("FICO Score", 300, 850, 720)
    region = st.selectbox("Region", ["North", "South", "Central", "North-East"])
    loan_purpose = st.selectbox("Purpose", ["p1", "p2", "p3", "p4"])
    
    payload_extras = {"Gender": "Male", "loan_type": "type1", "credit_type": "CIB", "age": "35-44", "dtir1": 40.0}
    
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("RUN RISK ASSESSMENT", type="primary", use_container_width=True)

# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------
c1, c2 = st.columns([3, 1])
with c1:
    st.title("AuraFin: Hybrid XGBoost/PyTorch System with Finacial Threshold Optimization")
    st.markdown("Cost-Sensitive Decision Engine • Optimized for Net Profitability")
with c2:
    st.markdown("")
    st.markdown('<div style="text-align: right; color: #10B981; font-weight: 600;">● SYSTEM ONLINE</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: right; color: #64748B; font-size: 14px;">Threshold: 0.59</div>', unsafe_allow_html=True)

st.markdown("---")

# TABS
tab_live, tab_impact, tab_research = st.tabs([
    " LIVE DECISION", 
    " BUSINESS IMPACT", 
    " METHODOLOGY"
])

# =========================================================
# TAB 1: LIVE DECISION (Fixed Rendering)
# =========================================================
with tab_live:
    if run_btn:
        payload = {
            "loan_amount": loan_amt, "income": income, "property_value": prop_val,
            "Credit_Score": credit_score, "LTV": ltv, "Region": region,
            "loan_purpose": loan_purpose, **payload_extras
        }
        
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                res = response.json()
                decision = res['decision']
                prob = res['risk_score']
                exposure = res['financial_analysis']['risk_weighted_exposure']
                threshold = res['threshold_used']
                
                # --- 1. TOP ROW: VERDICT CARD & GAUGE CHART (Aligned with fixed heights) ---
                col_left, col_right = st.columns([1, 2])
                
                with col_left:
                    # Determine style
                    style_class = "verdict-approve" if decision == "APPROVE" else "verdict-reject"
                    icon = "✅" if decision == "APPROVE" else "🚫"
                    
                    st.markdown(f"""
                    <div class="verdict-box {style_class}">
                        <div class="verdict-title">AI RECOMMENDATION</div>
                        <h1 class="verdict-value">{decision}</h1>
                        <div style="margin-top: 10px; font-weight: 500;">{icon} Confidence: {100-(prob*100):.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_right:
                    # --- 2. GAUGE CHART (Visual Impact, adjusted height) ---
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "<b>Risk Probability Score</b>", 'font': {'size': 16, 'color': '#64748B'}},
                        delta = {'reference': 59, 'increasing': {'color': "#EF4444"}, 'decreasing': {'color': "#10B981"}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748B"},
                            'bar': {'color': "#1E293B"}, # Dark Pointer
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "#E2E8F0",
                            'steps': [
                                {'range': [0, 59], 'color': "#D1FAE5"},  # Safe Green
                                {'range': [59, 100], 'color': "#FEE2E2"}], # Danger Red
                            'threshold': {
                                'line': {'color': "#EF4444", 'width': 4},
                                'thickness': 0.75,
                                'value': 59}}))
                    
                    fig.update_layout(height=200, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor="rgba(0,0,0,0)", font={'family': "Inter"})
                    st.plotly_chart(fig, use_container_width=True)

                # --- 3. METRICS ROW (Now as Cards, similar to Verdict) ---
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-title">CALCULATED RISK</div>
                        <div class="metric-value">{prob:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with m2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-title">FINANCIAL EXPOSURE</div>
                        <div class="metric-value">${exposure:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with m3:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-title">SAFETY THRESHOLD</div>
                        <div class="metric-value">{threshold:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # --- 4. EXPLAINABILITY (Fixed Rendering) ---
                st.markdown("---")
                st.subheader(" Logic Explanation")
                
                # Using columns for clean layout instead of HTML lists which can break
                exp_c1, exp_c2 = st.columns([1, 1])
                
                with exp_c1:
                    st.info(f"""
                    **The Verdict:**
                    The applicant's risk score is **{prob:.1%}**.
                    
                    This is **{'LOWER' if decision=='APPROVE' else 'HIGHER'}** than the optimized threshold of **{threshold}**, triggering the **{decision}** outcome.
                    """)
                    
                with exp_c2:
                    st.write("**Primary Risk Drivers:**")
                    
                    # Native Streamlit Markdown (Renders Perfectly)
                    st.markdown(f"""
                    - **LTV Ratio:** `{ltv:.1f}%` 
                      *Status:* {'🔴 High Leverage' if ltv > 80 else '🟢 Safe Equity'}
                    
                    - **Credit History:** `{credit_score}` 
                      *Status:* {'🔴 Subprime' if credit_score < 650 else '🟢 Prime'}
                    
                    - **LTI Ratio:** `{loan_amt/income:.1f}x` 
                      *Status:* Debt-to-Income load
                    """)

            else:
                st.error("Backend API Error")
        except Exception as e:
            st.error(f"Connection Failed: {e}")
    else:
        st.info("👈 Enter applicant details in the sidebar to generate a live risk assessment.")

# =========================================================
# TAB 2: FINANCIAL IMPACT
# =========================================================
with tab_impact:
    st.header("Financial Impact Analysis")
    
    col_i1, col_i2 = st.columns([1, 1])
    
    with col_i1:
        st.subheader("The $21.8 Million Advantage")
        st.markdown("""
        **1. The "Vanilla" Baseline (Red Bar):**
        Standard models optimize for *Accuracy*. On a dataset with 94% good loans, a standard model achieves 94% accuracy by simply predicting "No Default" for everyone.
        * **Result:** It captures **0%** of actual defaults.
        * **Financial Consequence:** The bank absorbs **$35.3 Million** in losses.
        
        **2. The AuraFin Solution (Green Bar):**
        We optimized for **Net Profit**. By implementing Cost-Sensitive Learning (15x penalty) and Threshold Optimization (0.59), we catch 71% of defaults while preserving safe customers.
        * **Result:** Minimizes loss to **$13.5 Million**.
        """)
        st.success(" **NET SAVINGS: $21.8 Million per batch**")
        
    with col_i2:
        if os.path.exists("images/11_true_business_impact.png"):
            st.image("images/11_true_business_impact.png", caption="Portfolio Loss Simulation", use_container_width=True)
        else:
            st.warning("Chart missing: images/11_true_business_impact.png")

# =========================================================
# TAB 3: METHODOLOGY
# =========================================================
with tab_research:
    st.markdown("## Engineering Methodology")
    st.markdown("A deep dive into the end-to-end data science lifecycle.")
    
    # 1. PROBLEM
    st.markdown("### 1. Problem Statement")
    st.markdown("Standard ML models optimize for accuracy. In credit risk, this is fatal because the cost of a False Negative (Default: \$60k) is 10x the cost of a False Positive (Rejection: \$6k).")
    
    st.markdown("---")
    
    # 2. FORENSICS
    st.markdown("### 2. Data Forensics & Cleaning")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Leakage Detection:** We identified that `rate_of_interest` was missing for 99% of rejected loans. This confirms it is a post-approval variable. We removed it to prevent Data Leakage.")
    with c2:
        if os.path.exists("images/01_leakage_evidence.png"):
            st.image("images/01_leakage_evidence.png", use_container_width=True, caption="Leakage Evidence")
    
    st.markdown("---")
    
    # 3. ENGINEERING
    st.markdown("### 3. Strategic Feature Engineering")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**'Skin in the Game' Theory:** We validated that Loan-to-Value (LTV) is the strongest predictor of default. High LTV (>80%) correlates with strategic default.")
    with c4:
        if os.path.exists("images/05_ltv_separation.png"):
            st.image("images/05_ltv_separation.png", use_container_width=True, caption="LTV Separation")

    st.markdown("---")
    
    # 4. OPTIMIZATION
    st.markdown("### 4. Financial Optimization")
    st.markdown("**The Cost Function:** We solved for the optimal decision threshold using the P&L equation:")
    st.latex(r"J(t) = (FN \times \$60,000) + (FP \times \$6,000)")
    
    c5, c6 = st.columns(2)
    with c5:
        if os.path.exists("images/09_financial_optimization.png"):
            st.image("images/09_financial_optimization.png", use_column_width=True, caption="Cost Minimization Curve")
    with c6:
        st.markdown("**Why 0.59?** Our weighted model was aggressive (paranoid). The optimization loop found that raising the threshold to 0.59 recovered **$300k** in profit without increasing risk.")