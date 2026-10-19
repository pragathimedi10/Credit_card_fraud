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
  # The model expects the number of features it was trained on (V1-V28 + NormalizedAmount)
  num_features = len(model.feature_names_in_)

  if test_option.startswith("Simulate Legitimate"):
    # Normal transaction pattern (zeros/baseline)
    sample_data = np.zeros((1, num_features))
    sample_data[0, -1] = 25.0  # Normal small amount
  else:
    # Suspicious anomaly pattern (random high variance values)
    sample_data = np.random.randn(1, num_features) * 4
    sample_data[0, -1] = 4500.0  # High amount

  # Make prediction
  prediction = model.predict(sample_data)

  # Display result
  if prediction[0] == 1:
    st.error(
        "🚨 ALERT: This transaction is flagged as FRAUDULENT! (High Risk)"
    )
  else:
    st.success("✅ SAFE: This transaction is classified as LEGITIMATE.")