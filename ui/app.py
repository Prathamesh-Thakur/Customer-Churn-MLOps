import streamlit as st
import requests

# Set page config
st.set_page_config(page_title="Telco Churn AI", page_icon="📡", layout="centered")

st.title("📡 Autonomous Churn Predictor")
st.markdown("Enter customer details below. The backend AI will dynamically filter for the features it currently deems most important.")

# 1. Collect standard customer profile
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# 2. Pack the data into a dictionary (matching your original dataset column names exactly!)
customer_data = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "InternetService": internet_service,
    "PaymentMethod": payment_method
    # Add other standard columns here if your baseline dataset has them!
}

st.divider()

# 3. Send to FastAPI
if st.button("Predict Churn Risk", type="primary", use_container_width=True):
    with st.spinner("Analyzing profile against latest ML model..."):
        try:
            # Notice we use the Docker service name 'fastapi' and port '8000'
            # (If running outside Docker for testing, change to http://localhost:8000/predict)
            response = requests.post("http://fastapi:8000/predict", json = customer_data)
            response.raise_for_status()
            
            prediction = response.json().get("churn_prediction_default")
            prediction_prob = response.json().get("churn_probability")

            st.metric(label="Churn Probability", value=f"{prediction_prob * 100:.1f}%")
            st.progress(float(prediction_prob))
            
            if prediction == 1:
                st.error("🚨 HIGH RISK: This customer is likely to churn.")
            else:
                st.success("✅ LOW RISK: This customer is stable.")
                
        except Exception as e:
            st.warning(f"Failed to connect to backend API. Error: {e}")