import joblib
import numpy as np
import streamlit as st

# Load the trained model
model = joblib.load("fraud_model.joblib")

st.title("💳 Credit Card Fraud Detection System")
st.write(
    "This web app uses a Machine Learning Random Forest model to predict whether a financial transaction is fraudulent or legitimate."
)

# User options to test
test_option = st.selectbox(
    "Select a transaction simulation:",
    ["Simulate Legitimate Transaction", "Simulate Suspicious Transaction"],
)

if st.button("Predict Transaction Status"):
  num_features = len(model.feature_names_in_)

  if test_option.startswith("Simulate Legitimate"):
    # Generate subtle, normal-looking feature values instead of pure zeros
    np.random.seed(42)
    sample_data = np.random.normal(0, 0.5, size=(1, num_features))
    sample_data[0, -1] = 35.0  # Normal small transaction amount
  else:
    # Suspicious anomaly pattern
    np.random.seed(100)
    sample_data = np.random.randn(1, num_features) * 4.5
    sample_data[0, -1] = 6500.0  # Unusually high amount

  # Make prediction
  prediction = model.predict(sample_data)

  # Display result
  if prediction[0] == 1:
    st.error(
        "🚨 ALERT: This transaction is flagged as FRAUDULENT! (High Risk)"
    )
  else:
    st.success("✅ SAFE: This transaction is classified as LEGITIMATE.")