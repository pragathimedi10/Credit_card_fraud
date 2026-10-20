import joblib
import numpy as np
import streamlit as st

# Load the trained model
model = joblib.load("fraud_model.joblib")

st.title("💳 Credit Card Fraud Detection System")
st.write(
    "Test a transaction using the Machine Learning Random Forest model below."
)

# Create input controls for the user
transaction_type = st.radio(
    "Choose Test Mode:", ["Quick Simulation", "Custom Transaction"]
)

num_features = len(model.feature_names_in_)

if transaction_type == "Quick Simulation":
  test_option = st.selectbox(
    "Select a transaction simulation:",
    ["Simulate Legitimate Transaction", "Simulate Suspicious Transaction"],
  )

  if st.button("Predict Transaction Status"):
    if test_option.startswith("Simulate Legitimate"):
      # Force a controlled, low-risk feature profile
      sample_data = np.zeros((1, num_features))
      sample_data[0, -1] = 10.0  # Small amount
    else:
      # Force high anomaly values
      sample_data = np.ones((1, num_features)) * 5.0
      sample_data[0, -1] = 5000.0  # Huge amount

    prediction = model.predict(sample_data)

    if prediction[0] == 1:
      st.error(
          "🚨 ALERT: This transaction is flagged as FRAUDULENT! (High Risk)"
      )
    else:
      st.success("✅ SAFE: This transaction is classified as LEGITIMATE.")

else:
  st.subheader("Customize Transaction Parameters")
  amount = st.number_input("Transaction Amount ($)", value=25.0, min_value=0.0)

  if st.button("Analyze Custom Transaction"):
    # Build feature array based on amount input
    sample_data = np.zeros((1, num_features))
    sample_data[0, -1] = amount  # Put amount in the normalized/amount slot

    prediction = model.predict(sample_data)

    if prediction[0] == 1:
      st.error(
          "🚨 ALERT: This transaction is flagged as FRAUDULENT! (High Risk)"
      )
    else:
      st.success("✅ SAFE: This transaction is classified as LEGITIMATE.")