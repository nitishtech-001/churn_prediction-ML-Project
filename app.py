import streamlit as st
import pandas as pd
import joblib
import sys

# 1. Page Configuration
st.set_page_config(
    page_title="Employee Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# 2. Cache the Model Loading to Keep the App Fast
@st.cache_resource
def load_prediction_model():
    try:
        # Tries to load your saved model file
        return joblib.load('./Models/churn_prediction.pkl')
    except FileNotFoundError:
        st.error("⚠️ '../Models/churn_prediction.pkl' not found. Please place your saved model in this folder.")
        return None

model = load_prediction_model()

# 3. Main Dashboard Header Layout
st.title("📊 Employee Churn Analysis & Risk Prediction")
st.markdown("Enter employee parameters in the sidebar to run real-time attrition risk assessments.")
st.divider()

# 4. Sidebar Inputs (Customize these fields based on your actual DataFrame columns)
st.sidebar.header("👤 Employee Profile Input")

with st.sidebar.form(key='employee_form'):

    st.text("Default Year: 2025")
    st.text("CustomerId and Surname does not takes because it does not impact on the chrun customers")

    # AGE INPUT
    age = st.slider("Age", min_value=18, max_value=100, value=30)
    # CREDIT SCORE
    credit_score = st.slider("Credit Score", min_value=0,max_value=1000,value=450)
    # SALARY ESTIMATED
    salary = st.number_input("Annual Salary ($)", min_value=10000, max_value=300000, value=50000, step=1000)
    # TENURE
    tenure = st.slider("Tenure (Years at Company)", min_value=0, max_value=10, value=3)
    # BALANCE
    balance = st.number_input("Bank Balance", min_value=0, value=10000, step=500)
    # NUM OF PRODUCTS
    num_of_products = st.slider("No of Products consume",min_value=1, max_value=4, value=3)
    # HAS CREDIT CARD
    hascrcard = st.selectbox("Has Credit Card", options=["No","Yes"],key="creditcard")
    # GENDER
    gender = st.selectbox("Gender",options=["Male","Female"],key="gender")
    # IS ACTIVE MEMBER
    isactivemember = st.selectbox("Is active member", options=["No","Yes"],key="activemember")
    # GEOGRAPHY
    geography = st.selectbox("Select Country",options=["Spain","France","Germany"],key="country")
    
    # Submit button inside the form
    submit_button = st.form_submit_button(label="🚀 Run Assessment")


# 5. Core Prediction Logic
if model is not None:
    # Construct a DataFrame matching your model's
    input_data = pd.DataFrame([{
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': 1 if hascrcard=="yes" else 0,
        'IsActiveMember': 1 if isactivemember == 'yes' else 0,
        'EstimatedSalary': salary,
        'Geography_Germany': 1 if geography=='germany' else 0,
        'Geography_Spain': 1 if geography=='spain' else 0,
        'Gender_Male': 1 if gender=='Male' else 0,
    }])
    
    # Calculate predictions
    prob = model.predict_proba(input_data)[0][1]
    risk_percentage = float(prob * 100)
    
    # Determine risk category & styling parameters
    if risk_percentage < 35:
        risk_status = "Low Risk ✅"
        status_color = "green"
    elif risk_percentage < 70:
        risk_status = "Medium Risk ⚠️"
        status_color = "orange"
    else:
        risk_status = "High Risk 🚨"
        status_color = "red"

    # 6. Display Dashboard Metrics UI
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Analysis Summary")
        st.metric(label="Calculated Attrition Risk", value=f"{risk_percentage:.1f}%")
        st.markdown(f"### Classification: :{status_color}[**{risk_status}**]")
        
    with col2:
        st.subheader("Risk Visualisation")
        # Visual progress bar serving as a risk meter
        st.progress(risk_percentage / 100.0)
        
        # Actionable business insights text block
        if risk_percentage >= 70:
            st.error("💡 **Retention Action Needed:** This profile exhibits high turnover probability patterns. Consider conducting a stay interview, evaluating salary benchmarks, or reviewing commute difficulties.")
        elif risk_percentage >= 35:
            st.warning("💡 **Monitoring Recommended:** Employee shows moderate signs of attrition risk. Ensure standard engagement and check-ins are active.")
        else:
            st.success("💡 **Stable Profile:** Employee risk match factors are consistent with stable, high-retention profiles inside the dataset.")

else:
    st.info("💡 Please ensure your trained model file is named 'final_turnover_model.pkl' and matches your feature definitions.")
