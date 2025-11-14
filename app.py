import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config MUST be the first Streamlit command
st.set_page_config(page_title="Personalized Healthcare Recommendations", page_icon="💓", layout="wide")

# Load trained model and scaler
try:
    model = joblib.load("heart_model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    st.error("❌ Model files not found. Please ensure 'heart_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

# Custom CSS for better styling with dark text
st.markdown("""
<style>
    /* Global Background and Text Color */
    html, body, [class*="st-"], .stApp {
        background-color: #0e1117 !important;
        color: #ffffff !important;
    }

    /* Force all text elements to white */
    * {
        color: #ffffff !important;
    }

    /* Keep interactive input components visible */
    input, select, textarea, .stTextInput input, .stNumberInput input {
        color: #ffffff !important;
        background-color: #1e1e1e !important;
        border: 1px solid #444 !important;
        border-radius: 5px;
    }

    /* Maintain visibility of radio button labels (Male/Female) */
    div[role='radiogroup'] label > div:nth-child(2) {
        color: #ffffff !important;
    }

    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        color: #00b4d8 !important;
        text-align: center;
        font-weight: 700;
        margin-bottom: 2rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff !important;
        margin: 1.5rem 0 0.5rem 0;
    }

    /* Recommendation cards */
    .recommendation-card {
        background-color: #1a1d23 !important;
        color: #ffffff !important;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #00b4d8;
    }

    /* Risk boxes */
    .risk-high {
        background-color: #3b0a0a !important;
        border-left: 6px solid #ff4d4d !important;
        color: #ffffff !important;
        padding: 20px;
        border-radius: 10px;
    }

    .risk-medium {
        background-color: #3b2a00 !important;
        border-left: 6px solid #ffd60a !important;
        color: #ffffff !important;
        padding: 20px;
        border-radius: 10px;
    }

    .risk-low {
        background-color: #0f3b0a !important;
        border-left: 6px solid #2dd881 !important;
        color: #ffffff !important;
        padding: 20px;
        border-radius: 10px;
    }

    /* Health indicator badges */
    .health-indicator {
        padding: 5px 10px;
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .health-good {
        background-color: #0f5132 !important;
        border-left: 4px solid #2dd881 !important;
    }
    .health-warning {
        background-color: #3b2a00 !important;
        border-left: 4px solid #ffd60a !important;
    }
    .health-critical {
        background-color: #3b0a0a !important;
        border-left: 4px solid #ff4d4d !important;
    }

    /* Streamlit tabs */
    .stTabs [role="tablist"] {
        border-bottom: 2px solid #00b4d8 !important;
    }

    /* Adjust all text outputs (predictions, metrics, etc.) */
    .stMarkdown, .stText, .stAlert, .stSuccess, .stInfo, .stError, .stWarning {
        color: #ffffff !important;
    }

    /* Keep layout clean */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        align-items: stretch !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Make radio buttons and checkboxes visible */
div[role='radiogroup'] label > div:first-child,
div[role='checkbox'] label > div:first-child {
    background-color: #222 !important;  /* light dark base */
    border: 2px solid #00b4d8 !important;
    border-radius: 50% !important;
    width: 16px !important;
    height: 16px !important;
    margin-right: 8px !important;
}

/* When selected */
div[role='radiogroup'] label div[aria-checked="true"] > div:first-child,
div[role='checkbox'] label div[aria-checked="true"] > div:first-child {
    background-color: #00b4d8 !important;
    border-color: #00b4d8 !important;
}

/* Ensure radio text remains readable */
div[role='radiogroup'] label > div:nth-child(2),
div[role='checkbox'] label > div:nth-child(2) {
    color: #ffffff !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Center and enlarge intro description */
.intro-text {
    text-align: center;
    color: #f8f9fa;
    font-size: 18px;
    line-height: 1.6;
    font-weight: 500;
    margin-top: -10px;
    margin-bottom: 25px;
}

/* Center the tabs below it */
div[data-baseweb="tab-list"] {
    justify-content: center !important;
}

/* Style the tab labels */
div[data-baseweb="tab"] {
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    transition: all 0.25s ease-in-out;
    border-radius: 10px;
    padding: 6px 14px !important;
}

/* Hover effect for tabs */
div[data-baseweb="tab"]:hover {
    background-color: rgba(43, 140, 255, 0.25);
    transform: scale(1.08);
    color: #2b8cff !important;
    cursor: pointer;
}

/* Active tab styling */
div[data-baseweb="tab"][aria-selected="true"] {
    background-color: #2b8cff !important;
    color: white !important;
    transform: scale(1.1);
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">❤️ Personalized Healthcare Recommendations</div>', unsafe_allow_html=True)

st.markdown("""
This app predicts the likelihood of heart disease and provides personalized healthcare recommendations 
based on your medical indicators and lifestyle factors.
""")

# Initialize session state for results
if 'probability' not in st.session_state:
    st.session_state.probability = None
    st.session_state.risk_level = None
    st.session_state.risk_factors = None

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["📊 HEALTH ASSESSMENT", "💡 RECOMENDATIONS", "📈 HEALTH ANALYTICS"])

with tab1:
    st.header("Personal Health Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographic Information")
        age = st.slider("Age", min_value=18, max_value=100, value=45)
        sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
        sex_encoded = 1 if sex == "Male" else 0
        
        st.subheader("Lifestyle Factors")
        exercise = st.slider("Weekly Exercise Hours", 0, 20, 5)
        
        # Better formatted smoking selection
        st.write("**Smoking Status**")
        smoking = st.radio(
            "Select your smoking status:",
            ["Never", "Former", "Current"],
            horizontal=False,
            key="smoking_radio"
        )
        
        st.write("**Alcohol Consumption**")
        alcohol = st.selectbox(
            "Select your alcohol consumption level:",
            ["None", "Light", "Moderate", "Heavy"],
            key="alcohol_select"
        )
        
    with col2:
        st.subheader("Medical Indicators")
        
        st.write("**Chest Pain Type**")
        cp = st.selectbox(
            "Select chest pain type:",
            ["0: Typical Angina", "1: Atypical Angina", "2: Non-anginal Pain", "3: Asymptomatic"],
            key="cp_select"
        )
        cp_encoded = int(cp[0])
        
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
        
        st.write("**Fasting Blood Sugar**")
        fbs = st.radio(
            "Fasting Blood Sugar > 120 mg/dl:",
            ["No", "Yes"], 
            horizontal=True,
            key="fbs_radio"
        )
        fbs_encoded = 1 if fbs == "Yes" else 0
        
        st.write("**Resting ECG Results**")
        restecg = st.selectbox(
            "Select resting ECG results:",
            ["0: Normal", "1: ST-T Wave Abnormality", "2: Left Ventricular Hypertrophy"],
            key="restecg_select"
        )
        restecg_encoded = int(restecg[0])
    
    col3, col4 = st.columns(2)
    
    with col3:
        thalachh = st.slider("Maximum Heart Rate Achieved", 60, 250, 150)
        
        st.write("**Exercise Induced Angina**")
        exang = st.radio(
            "Exercise induced angina:",
            ["No", "Yes"], 
            horizontal=True,
            key="exang_radio"
        )
        exang_encoded = 1 if exang == "Yes" else 0
        
    with col4:
        oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 10.0, 1.0, step=0.1)
        
        st.write("**Slope of Peak Exercise ST Segment**")
        slope = st.selectbox(
            "Select slope:",
            ["0: Upsloping", "1: Flat", "2: Downsloping"],
            key="slope_select"
        )
        slope_encoded = int(slope[0])
        
        ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 0)
        
        st.write("**Thalassemia**")
        thal = st.selectbox(
            "Select thalassemia type:",
            ["0: Normal", "1: Fixed Defect", "2: Reversible Defect"],
            key="thal_select"
        )
        thal_encoded = int(thal[0])

    # BMI input with better spacing
    st.write("**Body Mass Index (BMI)**")
    bmi = st.number_input(
        "Enter your BMI:",
        min_value=15.0, 
        max_value=50.0, 
        value=25.0, 
        step=0.1,
        key="bmi_input"
    )
    
    # Prepare input data
    input_data = np.array([[age, sex_encoded, cp_encoded, trestbps, chol, fbs_encoded, 
                          restecg_encoded, thalachh, exang_encoded, oldpeak, 
                          slope_encoded, ca, thal_encoded]])

    # Calculate risk factors
    risk_factors = 0
    if age > 50: risk_factors += 1
    if trestbps > 140: risk_factors += 1
    if chol > 240: risk_factors += 1
    if bmi > 30: risk_factors += 1
    if smoking == "Current": risk_factors += 1
    if exercise < 3: risk_factors += 1

    # Add some space before the button
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔍 Analyze My Health", type="primary", use_container_width=True):
        with st.spinner("Analyzing your health data and generating recommendations..."):
            
            # Scale input and make prediction
            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)
            prediction_proba = model.predict_proba(scaled_input)
            
            probability = prediction_proba[0][1]  # Probability of heart disease
            
            # Store results in session state
            st.session_state.probability = probability
            st.session_state.risk_factors = risk_factors
            
            # Determine risk level
            if probability > 0.7 or risk_factors >= 4:
                st.session_state.risk_level = "HIGH"
            elif probability > 0.3 or risk_factors >= 2:
                st.session_state.risk_level = "MEDIUM"
            else:
                st.session_state.risk_level = "LOW"
            
            # Display results
            st.markdown("---")
            st.header("📋 Health Assessment Results")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                # Risk level display
                risk_class = f"risk-{st.session_state.risk_level.lower()}"
                emoji = "🚨" if st.session_state.risk_level == "HIGH" else "⚠️" if st.session_state.risk_level == "MEDIUM" else "✅"
                
                st.markdown(f"""
                <div class="{risk_class}">
                    <h3 style="color: #000000;">{emoji} Risk Level: {st.session_state.risk_level}</h3>
                    <h4 style="color: #000000;">Heart Disease Probability: {probability:.1%}</h4>
                    <p style="color: #000000;">Identified Risk Factors: {risk_factors} out of 6</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability gauge
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.barh([0], [probability], color='red' if probability > 0.5 else 'orange' if probability > 0.3 else 'green')
                ax.set_xlim(0, 1)
                ax.set_xlabel('Heart Disease Probability')
                ax.set_yticklabels([])
                ax.set_title('Risk Probability Gauge')
                st.pyplot(fig)
                
            with col_result2:
                st.subheader("Key Health Indicators")
                
                indicators = {
                    "Blood Pressure": "Normal" if trestbps < 120 else "Elevated" if trestbps < 130 else "High",
                    "Cholesterol": "Normal" if chol < 200 else "Borderline" if chol < 240 else "High",
                    "Max Heart Rate": "Good" if thalachh > 150 else "Average" if thalachh > 130 else "Low",
                    "BMI": "Normal" if 18.5 <= bmi <= 24.9 else "Overweight" if bmi <= 29.9 else "Obese",
                    "Exercise": "Active" if exercise >= 5 else "Moderate" if exercise >= 3 else "Sedentary",
                    "Smoking": "Non-smoker" if smoking == "Never" else "Former smoker" if smoking == "Former" else "Current smoker"
                }
                
                for indicator, status in indicators.items():
                    color = "🟢" if status in ["Normal", "Good", "Active", "Non-smoker"] else "🟡" if status in ["Average", "Moderate", "Borderline", "Overweight", "Former smoker"] else "🔴"
                    st.write(f"{color} **{indicator}:** {status}")

with tab2:
    st.header("💡 Personalized Recommendations")
    
    if st.session_state.probability is not None:
        # Generate recommendations based on risk level
        if st.session_state.risk_level == "HIGH":
            st.markdown("""
            <div class="risk-high">
            <h3 style="color: #000000;">🚨 High Priority Recommendations</h3>
            </div>
            """, unsafe_allow_html=True)
            
            recommendations = [
                "🩺 **Consult a cardiologist immediately** for comprehensive evaluation",
                "💊 **Discuss medication options** with your healthcare provider",
                "🏥 **Schedule diagnostic tests**: ECG, Stress Test, Echocardiogram",
                "📱 **Monitor vital signs daily**: Blood pressure and heart rate",
                "🚭 **Quit smoking immediately** and avoid secondhand smoke",
                "🥗 **Adopt strict heart-healthy diet**: Low sodium, low cholesterol",
                "🏃 **Start supervised exercise program** with medical clearance",
                "😴 **Ensure 7-8 hours of quality sleep** nightly",
                "⚖️ **Achieve and maintain healthy weight** (BMI 18.5-24.9)"
            ]
            
        elif st.session_state.risk_level == "MEDIUM":
            st.markdown("""
            <div class="risk-medium">
            <h3 style="color: #000000;">⚠️ Moderate Risk Recommendations</h3>
            </div>
            """, unsafe_allow_html=True)
            
            recommendations = [
                "🩺 **Regular check-ups** with primary care physician every 6 months",
                "🏃 **Moderate exercise** 30 minutes daily, 5 days/week",
                "🥗 **Heart-healthy diet**: Focus on fruits, vegetables, whole grains",
                "⚖️ **Weight management** through balanced diet and exercise",
                "🚭 **Smoking cessation** program if applicable",
                "🍷 **Limit alcohol** to moderate levels",
                "😊 **Stress management**: Meditation, yoga, or relaxation techniques",
                "📊 **Monitor health metrics** regularly",
                "💤 **Quality sleep** 7-8 hours per night"
            ]
            
        else:
            st.markdown("""
            <div class="risk-low">
            <h3 style="color: #000000;">✅ Low Risk Maintenance</h3>
            </div>
            """, unsafe_allow_html=True)
            
            recommendations = [
                "✅ **Continue current healthy lifestyle** habits",
                "🏃 **Maintain regular physical activity** routine",
                "🥗 **Balanced nutrition** with variety of whole foods",
                "🩺 **Annual health check-ups** for prevention",
                "😊 **Stress management** and work-life balance",
                "💤 **Consistent sleep schedule**",
                "🚭 **Avoid smoking** and limit alcohol",
                "📚 **Stay informed** about heart health",
                "🎯 **Set health goals** for continuous improvement"
            ]
        
        # Display recommendations
        st.subheader("📝 Your Personalized Action Plan")
        for i, recommendation in enumerate(recommendations, 1):
            st.markdown(f'<div class="recommendation-card" style="color: #000000;">{i}. {recommendation}</div>', unsafe_allow_html=True)
        
        # Specific recommendations
        st.subheader("🔍 Specific Recommendations Based on Your Profile")
        
        specific_recommendations = []
        if trestbps > 140:
            specific_recommendations.append("**Blood Pressure**: Reduce sodium intake, monitor BP daily")
        if chol > 240:
            specific_recommendations.append("**Cholesterol**: Increase soluble fiber, reduce saturated fats")
        if bmi > 30:
            specific_recommendations.append("**Weight**: Aim for 5-10% weight loss through diet and exercise")
        if exercise < 3:
            specific_recommendations.append("**Activity**: Gradually increase to 150 minutes of moderate exercise weekly")
        if smoking == "Current":
            specific_recommendations.append("**Smoking**: Seek smoking cessation support immediately")
        if alcohol == "Heavy":
            specific_recommendations.append("**Alcohol**: Reduce alcohol consumption to moderate levels")
        
        if specific_recommendations:
            for rec in specific_recommendations:
                st.write(f"• {rec}")
        else:
            st.write("• No specific additional recommendations - keep up the good work!")
            
    else:
        st.info("Please complete the health assessment in the first tab to get personalized recommendations.")

with tab3:
    st.header("📈 Health Analytics & Insights")
    
    if st.session_state.probability is not None:
        col_anal1, col_anal2 = st.columns(2)
        
        with col_anal1:
            st.subheader("Risk Factor Analysis")
            
            # Create risk factors chart
            risk_data = {
                'Factor': ['Age > 50', 'BP > 140', 'Chol > 240', 'BMI > 30', 'Smoking', 'Low Exercise'],
                'Risk Present': [age > 50, trestbps > 140, chol > 240, bmi > 30, smoking == "Current", exercise < 3]
            }
            risk_df = pd.DataFrame(risk_data)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['red' if present else 'green' for present in risk_df['Risk Present']]
            ax.barh(risk_df['Factor'], [1]*len(risk_df), color=colors)
            ax.set_xlabel('Risk Impact')
            ax.set_title('Your Risk Factors Analysis')
            ax.set_xlim(0, 1)
            st.pyplot(fig)
        
        with col_anal2:
            st.subheader("Health Score Comparison")
            
            # Calculate health score
            health_score = 100 - (st.session_state.risk_factors * 15) - (st.session_state.probability * 20)
            health_score = max(0, min(100, health_score))
            
            st.metric("Overall Health Score", f"{health_score:.0f}/100")
            
            if health_score >= 80:
                st.success("Excellent health status!")
            elif health_score >= 60:
                st.warning("Good health with some areas for improvement")
            else:
                st.error("Needs significant health improvements")
        
        # Progress tracking
        st.subheader("📊 Health Improvement Tracker")
        st.info("Set goals and track your progress over time")
        
        col_track1, col_track2, col_track3 = st.columns(3)
        
        with col_track1:
            st.write("**Target Blood Pressure**")
            target_bp = st.number_input("Target BP (mm Hg)", 80, 200, 120, key="target_bp")
        with col_track2:
            st.write("**Target Cholesterol**")
            target_chol = st.number_input("Target Cholesterol", 100, 600, 200, key="target_chol")
        with col_track3:
            st.write("**Target BMI**")
            target_bmi = st.number_input("Target BMI", 15.0, 50.0, 22.0, step=0.1, key="target_bmi")
        
        if st.button("Set Health Goals", key="set_goals"):
            st.success("Health goals set! Track your progress regularly.")
    
    else:
        st.info("Complete the health assessment to see your analytics and insights.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #000000;'>
    <p><em>Disclaimer: This tool provides educational insights only. Always consult healthcare professionals for medical advice.</em></p>
    <p>Built with ❤️ for better heart health</p>
</div>
""", unsafe_allow_html=True)
