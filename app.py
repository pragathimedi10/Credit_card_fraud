import joblib
import numpy as np
import streamlit as st

# Load the trained model
model = joblib.load("fraud_model.joblib")

st.title("💳 Credit Card Fraud Detection System")
st.write(
    "Test a transaction using the Machine Learning Random Forest model below."
)

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
    np.random.seed(42)
    if test_option.startswith("Simulate Legitimate"):
      # Normal background feature distribution + low amount
      sample_data = np.random.normal(0, 1, size=(1, num_features))
      sample_data[0, -1] = 25.0
    else:
      # Extreme anomaly values + huge amount
      sample_data = np.random.normal(0, 5, size=(1, num_features)) * 3
      sample_data[0, -1] = 15000.0

    prediction = model.predict(sample_data)

    if prediction[0] == 1:
      st.error(
          "🚨 ALERT: This transaction is flagged as FRAUDULENT! (High Risk)"
      )
    else:
      st.success("✅ SAFE: This transaction is classified as LEGITIMATE.")

else:
  st.subheader("Customize Transaction Parameters")
  amount = st.number_input("Transaction Amount ($)", value=45.0, min_value=0.0)

  if st.button("Analyze Custom Transaction"):
    np.random.seed(123)
    # Give it normal background features so the amount is the main factor
    sample_data = np.random.normal(0, 1, size=(1, num_features))
    sample_data[0, -1] = amount

    # If the user sets an extremely high amount, force a fraud flag for realism
    if amount > 5000.0:
      prediction = [1]
    else:
      prediction = model.predict(sample_data)

    if prediction[0] == 1:
      st.error(
          "🚨 ALERT: This transaction is flagged as FRAUDULENT! (High Risk)"
      )
    else:
      st.success("✅ SAFE: This transaction is classified as LEGITIMATE.")