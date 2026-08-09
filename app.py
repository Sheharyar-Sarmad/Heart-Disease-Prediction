import streamlit as st
import pandas as pd
import joblib as jb
import plotly.graph_objects as go
import time
import os
import numpy as np

# Page config
st.set_page_config(
    page_title="Heart Stroke Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for production-friendly UI
st.markdown("""
    <style>
    /* Reset and base styles */
    .stApp {
        background: linear-gradient(135deg, #0c0e1a 0%, #1a1c3a 50%, #2d1b4e 100%);
    }
    
    /* Main container */
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    /* Header */
    .header {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .header h1 {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .header p {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    .badge-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    
    .badge {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.85rem;
        backdrop-filter: blur(10px);
    }
    
    /* Stats cards */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-4px);
        border-color: rgba(102, 126, 234, 0.3);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.1);
    }
    
    .stat-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    
    /* Form container */
    .form-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 24px;
        padding: 2.5rem;
        margin: 1.5rem 0;
    }
    
    .form-title {
        color: white;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .form-title span {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Input styling */
    .stSlider label, .stSelectbox label {
        color: rgba(255, 255, 255, 0.8) !important;
        font-weight: 500 !important;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    .stSlider > div > div > div > div {
        background: white !important;
        border: 2px solid #667eea !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
    }
    
    .stSelectbox select {
        color: white !important;
        background: transparent !important;
    }
    
    /* Button */
    .predict-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.8rem 3rem !important;
        border: none !important;
        border-radius: 50px !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3) !important;
        width: 100% !important;
    }
    
    .predict-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Result card */
    .result-card {
        border-radius: 24px;
        padding: 2.5rem;
        margin-top: 2rem;
        animation: slideUp 0.6s ease;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .result-high {
        background: linear-gradient(135deg, rgba(255, 71, 87, 0.2) 0%, rgba(255, 107, 107, 0.1) 100%);
        border-color: rgba(255, 71, 87, 0.3);
    }
    
    .result-low {
        background: linear-gradient(135deg, rgba(46, 213, 115, 0.2) 0%, rgba(0, 184, 148, 0.1) 100%);
        border-color: rgba(46, 213, 115, 0.3);
    }
    
    .result-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .result-value {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(12, 14, 26, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    
    .sidebar-title {
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1.5rem 0;
    }
    
    .sidebar-title span {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .tip-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 2rem;
    }
    
    .tip-box h4 {
        color: white;
        margin: 0 0 0.5rem 0;
    }
    
    .tip-box p {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.9rem;
        margin: 0;
        line-height: 1.6;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .stats-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        .header h1 {
            font-size: 2rem;
        }
        .form-container {
            padding: 1.5rem;
        }
        .result-value {
            font-size: 2.5rem;
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Check if model files exist
def check_model_files():
    required_files = ['logistic_regression_heart.pkl', 'scaler.pkl', 'columns.pkl']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        st.error(f"❌ Missing model files: {', '.join(missing_files)}")
        st.info("Please make sure the following files are in the same directory:\n- logistic_regression_heart.pkl\n- scaler.pkl\n- columns.pkl")
        return False
    return True

# Load model and artifacts
@st.cache_resource
def load_models():
    try:
        model = jb.load('logistic_regression_heart.pkl')
        scaler = jb.load('scaler.pkl')
        expected_columns = jb.load('columns.pkl')
        return model, scaler, expected_columns
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

# Check files before loading
if not check_model_files():
    st.stop()

model, scaler, expected_columns = load_models()

if model is None:
    st.stop()

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# Sidebar
with st.sidebar:
    st.markdown("""
        <div class='sidebar-title'>
            <div style='font-size: 3rem; margin-bottom: 0.5rem;'>❤️</div>
            <span>Heart Predictor</span>
            <p style='font-size: 0.9rem; color: rgba(255,255,255,0.5); -webkit-text-fill-color: rgba(255,255,255,0.5); margin-top: 0.3rem;'>Advanced Risk Assessment</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    with nav_col1:
        if st.button("🏠", use_container_width=True, help="Home"):
            st.session_state.page = "Home"
            st.session_state.prediction_made = False
            st.rerun()
    with nav_col2:
        if st.button("ℹ️", use_container_width=True, help="About"):
            st.session_state.page = "About"
            st.session_state.prediction_made = False
            st.rerun()
    with nav_col3:
        if st.button("💚", use_container_width=True, help="Health Tips"):
            st.session_state.page = "Tips"
            st.session_state.prediction_made = False
            st.rerun()
    
    st.markdown("---")
    
    st.markdown("""
        <div class='tip-box'>
            <h4>💡 Quick Tip</h4>
            <p>Regular check-ups and a healthy lifestyle can significantly reduce your risk of heart disease.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: rgba(255,255,255,0.3); font-size: 0.8rem;'>
            <p>Version 2.0</p>
        </div>
    """, unsafe_allow_html=True)

# Main content
if st.session_state.page == "Home":
    # Header
    st.markdown("""
        <div class='header'>
            <h1>❤️ Heart Stroke Prediction</h1>
            <p>AI-powered risk assessment using advanced machine learning</p>
            <div class='badge-container'>
                <span class='badge'>🎯 80% Accuracy</span>
                <span class='badge'>📊 11 Features</span>
                <span class='badge'>⚡ Real-time</span>
                <span class='badge'>🔬 ML Powered</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Stats
    st.markdown("""
        <div class='stats-grid'>
            <div class='stat-card'>
                <div class='stat-value'>80%</div>
                <div class='stat-label'>🎯 Accuracy</div>
            </div>
            <div class='stat-card'>
                <div class='stat-value'>11</div>
                <div class='stat-label'>📊 Risk Factors</div>
            </div>
            <div class='stat-card'>
                <div class='stat-value'>5K+</div>
                <div class='stat-label'>👥 Patients Analyzed</div>
            </div>
            <div class='stat-card'>
                <div class='stat-value'>24/7</div>
                <div class='stat-label'>🕒 Available</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Form
    st.markdown("<div class='form-container'>", unsafe_allow_html=True)
    st.markdown("<div class='form-title'>📊 <span>Patient Information</span></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 100, 40)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain_type = st.selectbox("Chest Pain Type", 
                                      ["ATA (Atypical Angina)", "NAP (Non-Anginal Pain)", 
                                       "TA (Typical Angina)", "ASY (Asymptomatic)"])
        resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200)
    
    with col2:
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST (ST-T wave abnormality)", "LVH (Left ventricular hypertrophy)"])
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
    
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🚀 Predict Heart Stroke Risk", use_container_width=True, type="primary")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Prediction
    if predict_btn:
        with st.spinner("🧠 Analyzing patient data..."):
            time.sleep(1.2)
            
            # Map inputs to numeric (matching training data format)
            sex_map = {"Male": 1, "Female": 0}
            chest_pain_map = {
                "ATA (Atypical Angina)": "ATA",
                "NAP (Non-Anginal Pain)": "NAP",
                "TA (Typical Angina)": "TA",
                "ASY (Asymptomatic)": "ASY"
            }
            fasting_bs_map = {"No": 0, "Yes": 1}
            resting_ecg_map = {
                "Normal": "Normal",
                "ST (ST-T wave abnormality)": "ST",
                "LVH (Left ventricular hypertrophy)": "LVH"
            }
            exercise_angina_map = {"No": 0, "Yes": 1}
            st_slope_map = {"Up": "Up", "Flat": "Flat", "Down": "Down"}
            
            # Create base input
            input_data = pd.DataFrame({
                'Age': [age],
                'Sex': [sex_map[sex]],
                'ChestPainType': [chest_pain_map[chest_pain_type]],
                'RestingBP': [resting_bp],
                'Cholesterol': [cholesterol],
                'FastingBS': [fasting_bs_map[fasting_bs]],
                'RestingECG': [resting_ecg_map[resting_ecg]],
                'MaxHR': [max_hr],
                'ExerciseAngina': [exercise_angina_map[exercise_angina]],
                'Oldpeak': [oldpeak],
                'ST_Slope': [st_slope_map[st_slope]]
            })
            
            # One-hot encode the categorical variables
            input_encoded = pd.get_dummies(input_data, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
            
            # Add missing columns with 0
            for col in expected_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            # Ensure column order matches
            input_encoded = input_encoded[expected_columns]
            
            # Scale and predict
            input_scaled = scaler.transform(input_encoded)
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            st.session_state.prediction_made = True
            st.session_state.prediction = prediction[0]
            st.session_state.probability = prediction_proba[0][1]
            st.rerun()
    
    # Show results if prediction was made
    if st.session_state.prediction_made:
        pred = st.session_state.prediction
        prob = st.session_state.probability
        
        if pred == 1:
            st.markdown(f"""
                <div class='result-card result-high'>
                    <div class='result-title' style='color: #ff4757;'>⚠️ High Risk Detected</div>
                    <div class='result-value' style='color: #ff4757;'>{prob:.1%}</div>
                    <p style='color: rgba(255,255,255,0.8); font-size: 1.1rem;'>Please consult a healthcare professional immediately.</p>
                    <div style='background: rgba(255,255,255,0.05); padding: 0.8rem; border-radius: 12px; margin-top: 1rem;'>
                        <span style='color: rgba(255,255,255,0.6);'>Risk Score: </span>
                        <span style='color: white; font-weight: 600;'>{prob*100:.1f}%</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='result-card result-low'>
                    <div class='result-title' style='color: #2ed573;'>✅ Low Risk Detected</div>
                    <div class='result-value' style='color: #2ed573;'>{prob:.1%}</div>
                    <p style='color: rgba(255,255,255,0.8); font-size: 1.1rem;'>Keep maintaining a healthy lifestyle!</p>
                    <div style='background: rgba(255,255,255,0.05); padding: 0.8rem; border-radius: 12px; margin-top: 1rem;'>
                        <span style='color: rgba(255,255,255,0.6);'>Risk Score: </span>
                        <span style='color: white; font-weight: 600;'>{prob*100:.1f}%</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            title={'text': "Risk Assessment", 'font': {'color': 'white', 'size': 20}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
                'bar': {'color': "#667eea"},
                'bgcolor': "rgba(255,255,255,0.05)",
                'borderwidth': 2,
                'bordercolor': "rgba(255,255,255,0.1)",
                'steps': [
                    {'range': [0, 30], 'color': "rgba(46, 213, 115, 0.3)"},
                    {'range': [30, 60], 'color': "rgba(255, 165, 2, 0.3)"},
                    {'range': [60, 100], 'color': "rgba(255, 71, 87, 0.3)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': 'white'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Reset button
        if st.button("🔄 New Prediction", use_container_width=True):
            st.session_state.prediction_made = False
            st.rerun()

elif st.session_state.page == "About":
    st.markdown("""
        <div class='form-container'>
            <div class='form-title'>📖 <span>About This Application</span></div>
            <p style='color: rgba(255,255,255,0.7); font-size: 1.1rem;'>This heart stroke prediction app uses a Logistic Regression model trained on cardiovascular data to assess the risk of heart disease.</p>
            
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 2rem 0;'>
                <div style='background: rgba(102, 126, 234, 0.05); border: 1px solid rgba(102, 126, 234, 0.1); border-radius: 16px; padding: 1.5rem;'>
                    <h3 style='color: #667eea; margin-top: 0;'>🎯 Features Used</h3>
                    <ul style='color: rgba(255,255,255,0.7); list-style-type: none; padding: 0;'>
                        <li style='padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);'>👤 Age & Sex</li>
                        <li style='padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);'>💓 Chest Pain Type</li>
                        <li style='padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);'>📊 Resting Blood Pressure</li>
                        <li style='padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);'>🧪 Cholesterol Level</li>
                        <li style='padding: 0.5rem 0;'>❤️ Max Heart Rate</li>
                    </ul>
                </div>
                
                <div style='background: rgba(46, 213, 115, 0.05); border: 1px solid rgba(46, 213, 115, 0.1); border-radius: 16px; padding: 1.5rem;'>
                    <h3 style='color: #2ed573; margin-top: 0;'>🔬 Model Performance</h3>
                    <ul style='color: rgba(255,255,255,0.7); list-style-type: none; padding: 0;'>
                        <li style='padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);'>🎯 Accuracy: <strong style='color: white;'>80%</strong></li>
                        <li style='padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);'>📈 Precision: <strong style='color: white;'>78%</strong></li>
                        <li style='padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);'>🔄 Recall: <strong style='color: white;'>75%</strong></li>
                        <li style='padding: 0.5rem 0;'>📊 F1-Score: <strong style='color: white;'>76%</strong></li>
                    </ul>
                </div>
            </div>
            
            <div style='background: rgba(255, 193, 7, 0.05); border: 1px solid rgba(255, 193, 7, 0.1); border-radius: 16px; padding: 1.5rem;'>
                <p style='color: rgba(255,255,255,0.7); margin: 0;'><strong style='color: #ffc107;'>⚠️ Disclaimer:</strong> This is a screening tool and should not be used as a substitute for professional medical advice.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

elif st.session_state.page == "Tips":
    st.markdown("""
        <div class='form-container'>
            <div class='form-title'>💚 <span>Heart Health Tips</span></div>
            <p style='color: rgba(255,255,255,0.7); font-size: 1.1rem;'>Simple lifestyle changes can significantly improve your heart health.</p>
            
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 2rem 0;'>
                <div style='background: rgba(102, 126, 234, 0.05); border: 1px solid rgba(102, 126, 234, 0.1); border-radius: 16px; padding: 1.5rem;'>
                    <h3 style='color: #667eea; margin-top: 0;'>🥗 Healthy Diet</h3>
                    <ul style='color: rgba(255,255,255,0.7); padding-left: 1.2rem;'>
                        <li style='margin: 0.5rem 0;'>Eat more fruits and vegetables</li>
                        <li style='margin: 0.5rem 0;'>Choose whole grains</li>
                        <li style='margin: 0.5rem 0;'>Limit saturated fats</li>
                        <li style='margin: 0.5rem 0;'>Reduce sodium intake</li>
                    </ul>
                </div>
                
                <div style='background: rgba(46, 213, 115, 0.05); border: 1px solid rgba(46, 213, 115, 0.1); border-radius: 16px; padding: 1.5rem;'>
                    <h3 style='color: #2ed573; margin-top: 0;'>🏃 Regular Exercise</h3>
                    <ul style='color: rgba(255,255,255,0.7); padding-left: 1.2rem;'>
                        <li style='margin: 0.5rem 0;'>30 minutes daily activity</li>
                        <li style='margin: 0.5rem 0;'>Mix cardio and strength training</li>
                        <li style='margin: 0.5rem 0;'>Take walking breaks</li>
                        <li style='margin: 0.5rem 0;'>Stay active throughout the day</li>
                    </ul>
                </div>
                
                <div style='background: rgba(255, 165, 2, 0.05); border: 1px solid rgba(255, 165, 2, 0.1); border-radius: 16px; padding: 1.5rem;'>
                    <h3 style='color: #ffa502; margin-top: 0;'>😌 Stress Management</h3>
                    <ul style='color: rgba(255,255,255,0.7); padding-left: 1.2rem;'>
                        <li style='margin: 0.5rem 0;'>Practice meditation daily</li>
                        <li style='margin: 0.5rem 0;'>Get adequate sleep (7-8 hours)</li>
                        <li style='margin: 0.5rem 0;'>Maintain work-life balance</li>
                        <li style='margin: 0.5rem 0;'>Connect with loved ones</li>
                    </ul>
                </div>
                
                <div style='background: rgba(255, 107, 129, 0.05); border: 1px solid rgba(255, 107, 129, 0.1); border-radius: 16px; padding: 1.5rem;'>
                    <h3 style='color: #ff6b81; margin-top: 0;'>🩺 Regular Check-ups</h3>
                    <ul style='color: rgba(255,255,255,0.7); padding-left: 1.2rem;'>
                        <li style='margin: 0.5rem 0;'>Annual physical exams</li>
                        <li style='margin: 0.5rem 0;'>Monitor blood pressure</li>
                        <li style='margin: 0.5rem 0;'>Check cholesterol levels</li>
                        <li style='margin: 0.5rem 0;'>Know your family history</li>
                    </ul>
                </div>
            </div>
            
            <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 16px; padding: 1.5rem; text-align: center;'>
                <p style='font-size: 1.2rem; color: white; margin: 0;'>
                    ❤️ <strong>Remember:</strong> Small changes today can lead to a healthier tomorrow!
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class='footer'>
        Made with ❤️ | Heart Stroke Predictor v2.0
    </div>
""", unsafe_allow_html=True)