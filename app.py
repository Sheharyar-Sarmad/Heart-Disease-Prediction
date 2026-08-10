import streamlit as st
import pandas as pd
import joblib as jb
import plotly.graph_objects as go
import time
import os
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# PAGE CONFIG (Fixed page_icon to an emoji instead of local path)
st.set_page_config(
    page_title="Heart Predictor",
    page_icon="./assets/logo_icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM GRADIENT FAVICON 
st.markdown("""
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Cdefs%3E%3ClinearGradient id='g' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' stop-color='%23667eea'/%3E%3Cstop offset='100%25' stop-color='%23764ba2'/%3E%3C/linearGradient%3E%3C/defs%3E%3Cpath d='M50 90C50 90 20 70 10 50C0 30 10 10 30 10C45 10 50 25 50 25C50 25 55 10 70 10C90 10 100 30 90 50C80 70 50 90 50 90Z' fill='url(%23g)'/%3E%3Ctext x='50' y='65' font-family='Arial' font-size='35' font-weight='bold' fill='white' text-anchor='middle'%3E%3Ctspan dy='-10'%3E%F0%9F%92%93%3C/tspan%3E%3C/text%3E%3C/svg%3E">
""", unsafe_allow_html=True)

# GROQ AI SETUP 
GROQ_MODEL = "llama-3.3-70b-versatile"

try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
except Exception:
    GROQ_API_KEY = None
if not GROQ_API_KEY:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def init_groq_client():
    return Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

def get_ai_response(prediction_result, risk_score, patient_data):
    if not GROQ_API_KEY: return "AI insights unavailable (API key missing)"
    client = init_groq_client()
    if not client: return "AI service error"

    risk_level = "HIGH" if prediction_result == 1 else "LOW"
    patient_summary = "\n".join(f"- {k}: {v}" for k, v in patient_data.items())

    system_prompt = """You are a compassionate, professional healthcare AI assistant specializing in heart health.
    Provide helpful, empathetic, and medically accurate advice based on the patient's risk assessment.
    Keep responses concise (2-3 short paragraphs) and actionable. Use plain, warm language, avoid jargon.
    Always end with a one-line disclaimer that this is not a substitute for professional medical advice.
    If risk is HIGH: focus on urgent, concrete next steps and when to see a doctor.
    If risk is LOW: focus on prevention and positive reinforcement."""

    user_prompt = f"""Patient data:
{patient_summary}

Prediction: {risk_level} RISK
Risk Score: {risk_score:.1f}%

Please give:
1. A brief, plain-language assessment
2. 3-4 concrete recommendations
3. Clear guidance on next steps"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"__ERROR__::{str(e)}"

def ask_followup(question):
    client = init_groq_client()
    if not client: return "AI service isn't configured right now."
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful heart health AI assistant. Give concise, accurate, empathetic answers in plain language."},
                {"role": "user", "content": question},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I ran into an error reaching the AI service: {e}"

# DASHBOARD UI CSS (FIXED SIDEBAR TOGGLE & REMOVED HIDDEN HEADER)
CUSTOM_CSS = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
    .stApp { background: radial-gradient(circle at 50% 0%, #161c3a 0%, #060612 70%); }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: rgba(12,14,26,0.5); }
    ::-webkit-scrollbar-thumb { background: #8b5cf6; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #764ba2; }
    section[data-testid="stSidebar"] {
        background: rgba(10, 11, 30, 0.8) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.2rem;
    }
    .stat-card { padding: 1.5rem; border-radius: 16px; position: relative; overflow: hidden; transition: all 0.3s ease; }
    .stat-card:hover { transform: translateY(-4px); }
    .stat-card .stat-icon { 
        display: inline-flex; align-items: center; justify-content: center;
        width: 50px; height: 50px; border-radius: 12px;
        font-size: 1.5rem; margin-bottom: 0.8rem;
    }
    .stat-card .stat-val { font-size: 2rem; font-weight: 800; color: white; }
    .stat-card .stat-label { font-size: 0.9rem; color: rgba(255,255,255,0.6); }
    
    .stat-blue { background: rgba(16, 185, 255, 0.1); border: 1px solid rgba(16, 185, 255, 0.3); }
    .stat-blue .stat-icon { background: rgba(16, 185, 255, 0.2); color: #10b9ff; box-shadow: 0 0 15px rgba(16, 185, 255, 0.2); }
    .stat-purple { background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.3); }
    .stat-purple .stat-icon { background: rgba(139, 92, 246, 0.2); color: #8b5cf6; box-shadow: 0 0 15px rgba(139, 92, 246, 0.2); }
    .stat-pink { background: rgba(236, 72, 153, 0.1); border: 1px solid rgba(236, 72, 153, 0.3); }
    .stat-pink .stat-icon { background: rgba(236, 72, 153, 0.2); color: #ec4899; box-shadow: 0 0 15px rgba(236, 72, 153, 0.2); }
    .stat-orange { background: rgba(251, 146, 60, 0.1); border: 1px solid rgba(251, 146, 60, 0.3); }
    .stat-orange .stat-icon { background: rgba(251, 146, 60, 0.2); color: #fb923c; box-shadow: 0 0 15px rgba(251, 146, 60, 0.2); }

    .dash-title { font-size: 1.8rem; font-weight: 700; background: linear-gradient(135deg, #ffffff, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }
    .dash-subtitle { color: rgba(255,255,255,0.6); font-size: 0.95rem; margin-top: 0.2rem; }
    .card-header { font-size: 1.2rem; font-weight: 600; color: white; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 0.6rem; }
    
    /* ===== UPDATED BUTTON STYLES FOR PERFECT UNIFORM HEIGHT ===== */
    .stButton > button {
        width: 100%;
        min-height: 85px; /* Forces all buttons to be exactly the same height */
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        background: linear-gradient(135deg, #8b5cf6, #ec4899);
        border: none;
        color: white;
        font-weight: 700;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
        white-space: normal; /* Allows wrapping while staying perfectly centered */
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 30px rgba(139, 92, 246, 0.5);
    }
    
    .stSelectbox > div > div, .stSlider > div > div {
        background: rgba(255, 255, 255, 0.05) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important; color: white !important;
    }
    .stSelectbox > div > div:focus-within { border-color: #8b5cf6 !important; box-shadow: 0 0 15px rgba(139, 92, 246, 0.2); }
    .stSlider label, .stSelectbox label { color: rgba(255,255,255,0.7) !important; font-size: 0.85rem !important; font-weight: 500 !important; }
    .datetime-widget {
        background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px; padding: 0.5rem 1rem; color: rgba(255,255,255,0.7);
        font-size: 0.8rem; text-align: center; float: right; backdrop-filter: blur(10px);
    }
    .datetime-widget i { margin-right: 0.3rem; }
    .sidebar-logo { text-align: center; padding: 1.5rem 0; }
    .sidebar-logo i { font-size: 3.5rem; background: linear-gradient(135deg, #ec4899, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: block; margin-bottom: 0.8rem; }
    .sidebar-logo span { font-size: 1.4rem; font-weight: 700; background: linear-gradient(135deg, #ffffff, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .sidebar-logo p { font-size: 0.75rem; color: rgba(255,255,255,0.5); margin-top: 0.2rem; -webkit-text-fill-color: rgba(255,255,255,0.5); }
    
    /* ===== FIXED CSS: Hides deploy menu but KEEPS the sidebar toggle ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    h1 > a, h2 > a { display: none !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# MODEL LOADING 
REAL_ACCURACY = 86.96

def check_model_files():
    required_files = ['logistic_regression_heart.pkl', 'scaler.pkl', 'columns.pkl']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        st.error(f"❌ Missing model files: {', '.join(missing_files)}")
        return False
    return True

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

if not check_model_files(): st.stop()
model, scaler, expected_columns = load_models()
if model is None: st.stop()

# SESSION STATE
defaults = {
    'page': "Home", 'prediction_made': False,
    'prediction': None, 'probability': None, 'patient_data': None, 'predicted_at': None,
    'ai_response': None, 'ai_error': None, 'chat_history': [], 'preset': None,
    'predictions_counter': 0,
    'healthy_predictions_counter': 0,
    'prediction_times': [],
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

PRESETS = {
    "Healthy example": dict(age=32, sex="Female", cpt="ATA", bp=110, chol=170, fbs="No", ecg="Normal", hr=175, angina="No", oldpeak=0.2, slope="Up"),
    "Higher-risk example": dict(age=63, sex="Male", cpt="ASY", bp=155, chol=280, fbs="Yes", ecg="ST", hr=110, angina="Yes", oldpeak=2.8, slope="Flat"),
}

# SIDEBAR 
with st.sidebar:
    st.markdown("""
        <div class='sidebar-logo'>
            <i class="fas fa-heartbeat"></i>
            <span>Heart Predictor</span>
            <p>AI-Powered Heart Disease Prediction</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    pages = [("Home", "🏠 Home"), ("About", "ℹ️ About"), ("AI Chat", "🤖 AI Chat")]
    for key, label in pages:
        is_active = st.session_state.page == key
        if st.button(label, width="stretch", type="primary" if is_active else "secondary", key=f"btn_{key}"):
            st.session_state.page = key
            st.rerun()

    st.markdown("---")
    if GROQ_API_KEY:
        st.markdown("""
            <div style='background: rgba(46,213,115,0.1); border: 1px solid rgba(46,213,115,0.3); border-radius: 12px; padding: 0.8rem 1rem; color: #2ed573; font-size: 0.85rem; display: flex; align-items: center; gap: 0.2rem;'>
                <i class="fas fa-circle" style="font-size: 0.5rem;"></i> <span style='color: white; font-weight: 500;'>AI</span> <span style='color: rgba(255,255,255,0.5); font-size: 0.7rem; margin-left: 1;'>Powered by Groq</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ AI Disabled")
        st.caption("Add `GROQ_API_KEY` to secrets.")

    st.markdown("---")
    st.markdown("""
        <div style='font-size: 0.8rem; color: rgba(255,255,255,0.3); text-align: center;'>
            <i class="fas fa-heart" style="color: #ff4757; margin-right: 0.3rem;"></i> "Take care of your heart today."
            <br><br> © 2024 Heart Predictor
        </div>
    """, unsafe_allow_html=True)


def risk_color(prob):
    if prob < 0.30: return "#2ed573"
    elif prob < 0.60: return "#ffa502"
    return "#ff4757"

def risk_label(prob):
    if prob < 0.30: return "LOW RISK"
    elif prob < 0.60: return "MODERATE RISK"
    return "HIGH RISK"

# HOME PAGE 
if st.session_state.page == "Home":
    now = datetime.now()
    st.markdown(f"""
        <div class='datetime-widget'>
            <i class="far fa-calendar-alt"></i> {now.strftime('%b %d, %Y')} &nbsp;|&nbsp; <i class="far fa-clock"></i> {now.strftime('%I:%M %p')}
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='clear:both;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='dash-title'>Welcome back! 👋</div>", unsafe_allow_html=True)
    st.markdown("<div class='dash-subtitle'>Get your heart health prediction with AI-powered insights.</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
    <div id="local-time-widget" class='datetime-widget'>
        <i class="far fa-calendar-alt"></i> <span id="date-text">Loading...</span> &nbsp;|&nbsp; <i class="far fa-clock"></i> <span id="time-text">Loading...</span>
    </div>
    <script>
        function updateLocalTime() {
            const now = new Date();
            const dateStr = now.toLocaleDateString('en-US', { month: 'short', day: '2-digit', year: 'numeric' });
            const timeStr = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true });
            document.getElementById('date-text').innerText = dateStr;
            document.getElementById('time-text').innerText = timeStr;
        }
        updateLocalTime();
        setInterval(updateLocalTime, 60000); // Updates automatically every minute
    </script>
       """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class='stat-card stat-purple'>
                <div class='stat-icon'><i class="fas fa-database"></i></div>
                <div class='stat-val'>918</div>
                <div class='stat-label'>Data Samples</div>
                <div style='color: rgba(255,255,255,0.4); font-size: 0.7rem; margin-top: 0.2rem;'>Standard Dataset</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class='stat-card stat-pink'>
                <div class='stat-icon'><i class="fas fa-clock"></i></div>
                <div class='stat-val'>24/7</div>
                <div class='stat-label'>Uptime Guarantee</div>
                <div style='color: rgba(255,255,255,0.4); font-size: 0.7rem; margin-top: 0.2rem;'>Always Available</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class='stat-card stat-orange'>
                <div class='stat-icon'><i class="fas fa-microchip"></i></div>
                <div class='stat-val'>Groq</div>
                <div class='stat-label'>AI Engine</div>
                <div style='color: rgba(255,255,255,0.4); font-size: 0.7rem; margin-top: 0.2rem;'>LLM Powered</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")
    with col_left:
        st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("<div class='card-header'><i class='fas fa-stethoscope' style='color: #8b5cf6;'></i> Make New Prediction</div>", unsafe_allow_html=True)
        st.markdown("<div style='color:rgba(255,255,255,0.5); font-size: 0.9rem; margin-bottom: 1.5rem;'>Fill in the patient details to get heart disease prediction.</div>", unsafe_allow_html=True)
        
        sp1, sp2, sp3 = st.columns([1, 1, 1])
        with sp1: 
            if st.button("💚 Healthy Example", use_container_width=True): st.session_state.preset = "Healthy example"; st.rerun()
        with sp2: 
            if st.button("⚠️ High Risk Ex.", use_container_width=True): st.session_state.preset = "Higher-risk example"; st.rerun()
        with sp3:
            if st.button("↺ Clear", use_container_width=True): st.session_state.preset = None; st.rerun()
            
        preset = PRESETS.get(st.session_state.preset, {})

        with st.form("dashboard_patient_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Age", 18, 100, preset.get("age", 40))
                chest_pain_type = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"], index=["ATA", "NAP", "TA", "ASY"].index(preset.get("cpt", "ATA")))
                cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, preset.get("chol", 200))
            with c2:
                sex = st.selectbox("Sex", ["Male", "Female"], index=["Male", "Female"].index(preset.get("sex", "Male")))
                resting_bp = st.slider("Resting BP (mm Hg)", 80, 200, preset.get("bp", 120))
                max_hr = st.slider("Max Heart Rate", 60, 220, preset.get("hr", 150))

            st.markdown("---")
            c3, c4 = st.columns(2)
            with c3:
                fasting_bs = st.selectbox("Fasting BS > 120 mg/dL", ["No", "Yes"], index=["No", "Yes"].index(preset.get("fbs", "No")))
                exercise_angina = st.selectbox("Exercise Angina", ["No", "Yes"], index=["No", "Yes"].index(preset.get("angina", "No")))
            with c4:
                resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], index=["Normal", "ST", "LVH"].index(preset.get("ecg", "Normal")))
                st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], index=["Up", "Flat", "Down"].index(preset.get("slope", "Up")))

            oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, preset.get("oldpeak", 1.0), step=0.1)
            predict_btn = st.form_submit_button("💓 Predict Heart Disease", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='glass-card' style='height: 100%; display: flex; flex-direction: column; justify-content: center;'>", unsafe_allow_html=True)
        st.markdown("<div class='card-header'><i class='fas fa-chart-pie' style='color: #10b9ff;'></i> Latest Prediction Result</div>", unsafe_allow_html=True)

        if predict_btn:
            progress = st.progress(0, text="Processing…")
            try:
                start_time = time.time()
                sex_map = {"Male": 1, "Female": 0}
                chest_pain_map = {"ATA": "ATA", "NAP": "NAP", "TA": "TA", "ASY": "ASY"}
                fasting_bs_map = {"No": 0, "Yes": 1}
                resting_ecg_map = {"Normal": "Normal", "ST": "ST", "LVH": "LVH"}
                exercise_angina_map = {"No": 0, "Yes": 1}
                st_slope_map = {"Up": "Up", "Flat": "Flat", "Down": "Down"}

                input_data = pd.DataFrame({
                    'Age': [age], 'Sex': [sex_map[sex]], 'ChestPainType': [chest_pain_map[chest_pain_type]],
                    'RestingBP': [resting_bp], 'Cholesterol': [cholesterol], 'FastingBS': [fasting_bs_map[fasting_bs]],
                    'RestingECG': [resting_ecg_map[resting_ecg]], 'MaxHR': [max_hr],
                    'ExerciseAngina': [exercise_angina_map[exercise_angina]], 'Oldpeak': [oldpeak],
                    'ST_Slope': [st_slope_map[st_slope]],
                })
                input_encoded = pd.get_dummies(input_data, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
                for col in expected_columns:
                    if col not in input_encoded.columns: input_encoded[col] = 0
                input_encoded = input_encoded[expected_columns]
                input_scaled = scaler.transform(input_encoded)
                
                prediction = model.predict(input_scaled)
                prediction_proba = model.predict_proba(input_scaled)
                
                st.session_state.predictions_counter += 1
                st.session_state.prediction_times.append(time.time() - start_time)
                if prediction[0] == 0: st.session_state.healthy_predictions_counter += 1
                
                st.session_state.prediction_made = True
                st.session_state.prediction = prediction[0]
                st.session_state.probability = prediction_proba[0][1]
                st.session_state.patient_data = {
                    'Age': age, 'Sex': sex, 'ChestPainType': chest_pain_type, 'RestingBP': resting_bp,
                    'Cholesterol': cholesterol, 'FastingBS': fasting_bs, 'RestingECG': resting_ecg,
                    'MaxHR': max_hr, 'ExerciseAngina': exercise_angina, 'Oldpeak': oldpeak, 'ST_Slope': st_slope,
                }
                st.session_state.predicted_at = datetime.now().strftime("%b %d, %Y %I:%M %p")

                progress.progress(70, "Generating AI Insights…")
                ai_response = get_ai_response(prediction[0], prediction_proba[0][1] * 100, st.session_state.patient_data)
                if ai_response and ai_response.startswith("__ERROR__::"): st.session_state.ai_error = ai_response.split("::", 1)[1]
                else: st.session_state.ai_response = ai_response
                
                progress.progress(100, "Done!")
                time.sleep(0.3)
                progress.empty()
                st.rerun()
            except Exception as e:
                progress.empty()
                st.error(f"Error processing prediction: {e}")

        if st.session_state.prediction_made:
            prob = st.session_state.probability
            pct = prob * 100
            res_color = risk_color(prob)
            res_label = risk_label(prob)

            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pct,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"<b>{res_label}</b>", 'font': {'size': 24, 'color': res_color}},
                number = {'suffix': "%", 'font': {'size': 30, 'color': 'white'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.2)", 'tickfont': {'color': 'rgba(255,255,255,0.2)'}},
                    'bar': {'color': res_color, 'thickness': 0.3},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(46,213,115,0.05)"},
                        {'range': [30, 60], 'color': "rgba(255,165,2,0.05)"},
                        {'range': [60, 100], 'color': "rgba(255,71,87,0.05)"}],
                    'threshold': {'line': {'color': res_color, 'width': 4}, 'thickness': 0.7, 'value': pct}
                }
            ))
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)", font={'color': 'white'})
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            safe_prob = min(1.0, max(0.0, prob))
            st.markdown(f"""
                <div style='display:flex; justify-content:space-between; color:rgba(255,255,255,0.6); font-size:0.8rem; margin-bottom:-0.5rem;'>
                    <span>Risk Score</span>
                    <span style='color: {res_color}; font-weight: bold;'>{pct:.1f}%</span>
                </div>
            """, unsafe_allow_html=True)
            st.progress(safe_prob, text=None)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔍 View Full Analysis", use_container_width=True, key="view_analysis"):
                st.session_state.page = "About"
                st.rerun()

        else:
            st.markdown("""
                <div style='text-align:center; padding: 4rem 0; color: rgba(255,255,255,0.4);'>
                    <i class="fas fa-heart" style='font-size: 4rem; display:block; margin-bottom: 1rem;'></i>
                    <p style='font-size: 1.1rem;'>Submit a prediction to see results here.</p>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    ai_col, chat_col = st.columns([1, 1], gap="large")
    with ai_col:
        st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("<div class='card-header'><i class='fas fa-sparkles' style='color: #ec4899;'></i> AI Health Insights</div>", unsafe_allow_html=True)
        
        if st.session_state.prediction_made:
            if st.session_state.ai_response:
                st.markdown(f"<p style='color: rgba(255,255,255,0.85); line-height: 1.6;'>{st.session_state.ai_response.replace(chr(10), '<br>')}</p>", unsafe_allow_html=True)
            elif st.session_state.ai_error:
                st.warning(f"AI insights temporarily unavailable. ({st.session_state.ai_error})")
            else:
                st.info("Generating insights...")
        else:
            st.markdown("""
                <div style='text-align:center; padding: 2rem 0; color: rgba(255,255,255,0.6);'>
                    <i class="fas fa-brain" style='font-size: 3rem; background: linear-gradient(135deg, #8b5cf6, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display:block; margin-bottom: 0.8rem;'></i>
                    <p>Great news! Your heart health indicators look good. Continue your healthy lifestyle with regular exercise and a balanced diet. Keep up the excellent work!</p>
                </div>
            """, unsafe_allow_html=True)
        
        if st.session_state.ai_error:
            if st.button("🔁 Retry AI Insights", use_container_width=True):
                with st.spinner("Retrying…"):
                    resp = get_ai_response(st.session_state.prediction, st.session_state.probability * 100, st.session_state.patient_data)
                    if resp and resp.startswith("__ERROR__::"): st.session_state.ai_error = resp.split("::", 1)[1]
                    else: st.session_state.ai_response = resp; st.session_state.ai_error = None
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with chat_col:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-header'><i class='fas fa-comment-dots' style='color: #10b9ff;'></i> Ask AI Assistant</div>", unsafe_allow_html=True)
        
        # UPDATED CHAT LOOP (Removed avatars to clean up UI) 
        for chat in st.session_state.chat_history[-2:]:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])

        if not GROQ_API_KEY:
            st.warning("⚠️ Groq AI not configured. Chat is disabled.")
        else:
            user_question = st.chat_input("Ask anything about heart health...")
            if user_question:
                with st.spinner("🤖 Thinking…"):
                    answer = ask_followup(user_question)
                    st.session_state.chat_history.append({"question": user_question, "answer": answer})
                st.rerun()

        st.markdown("<div style='font-size:0.7rem; color:rgba(255,255,255,0.3); margin-top: 1rem;'>⚡ Powered by Groq AI</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ABOUT PAGE
elif st.session_state.page == "About":
    st.title("ℹ️ About This Application")
    st.write("This app uses a Logistic Regression model trained on cardiovascular data to estimate the risk of heart disease.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### **🧠 Features Used**")
        st.write("- Age & Sex")
        st.write("- Chest Pain Type")
        st.write("- Resting Blood Pressure")
        st.write("- Cholesterol Level")
        st.write("- Max Heart Rate & more")
    with col2:
        st.markdown("### **📊 Model Performance**")
        st.write(f"- **Accuracy:** {REAL_ACCURACY}%")
        st.write("- **Precision:** 87%")
        st.write("- **Recall:** 87%")
        st.write("- **F1-Score:** 88.46%")

    st.warning("⚠️ **Disclaimer:** This is a screening tool built for education and demonstration. It is not a medical device and should never replace professional diagnosis or advice from a qualified clinician.")

# AI CHAT PAGE 
elif st.session_state.page == "AI Chat":
    st.title("💬 AI Health Assistant")
    st.write("Ask any questions about heart health, your results, or lifestyle recommendations.")
    
    if not GROQ_API_KEY:
        st.warning("⚠️ Groq AI is not configured yet.")
        st.info("To set up: Streamlit Cloud → Settings → Secrets → add `GROQ_API_KEY`.")
        st.stop()

    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
        
    st.markdown("---")
     
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            with st.chat_message("user"): st.write(chat["question"])
            with st.chat_message("assistant"): st.write(chat["answer"])
    else:
        st.info("Start a conversation about heart health. Ask about symptoms, prevention, diet, or exercise.")

    user_question = st.chat_input("Ask me anything about heart health…")
    if user_question:
        with st.spinner("🤖 Thinking…"):
            answer = ask_followup(user_question)
            st.session_state.chat_history.append({"question": user_question, "answer": answer})
        st.rerun()