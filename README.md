# 💳 ML Driven - Credit Card Fraud Detection System

An end-to-end Machine Learning web application designed to detect fraudulent financial transactions in real time. The system leverages a **Random Forest Classifier** trained on transaction data and features an interactive web dashboard built with **Streamlit** for instant risk assessment.

---
### Demo link https://creditcard-fraud-mlproject.streamlit.app/

## 📌 Features

* **Machine Learning Engine:** Utilizes a Random Forest algorithm to evaluate key transaction metrics and anonymized features ($V_1–V_{28}$).
* **Real-Time Risk Assessment:** Instantly classifies transactions into **✅ Legitimate (SAFE)** or **🚨 Fraudulent (High Risk)**.
* **Interactive Testing Modes:**
  * **Quick Simulation:** Allows recruiters and testers to run one-click simulations for standard vs. suspicious transactions.
  * **Custom Transaction Input:** Enables users to manually enter specific transaction amounts to evaluate prediction outputs.
* **Cloud-Deployed:** Fully hosted on **Streamlit Community Cloud** with continuous deployment linked directly to the GitHub repository.

---

## 🛠️ Tech Stack & Tools

* **Programming Language:** Python 3.10+
* **Machine Learning & Data Science:** Scikit-Learn, Pandas, NumPy, Joblib
* **Web Framework:** Streamlit
* **Version Control & Hosting:** Git, GitHub, Streamlit Cloud

---

## 📊 Dataset & Model Pipeline

1. **Preprocessing & Feature Engineering:**
   * Handled anonymized Principal Component Analysis (PCA) features ($V_1–V_{28}$) and normalized transaction amounts.
   * Addressed missing values and prepared structured feature matrices ($X$) and target vectors ($y$).
2. **Model Selection:**
   * Trained a **Random Forest Classifier** to handle complex non-linear patterns and maintain robustness against overfitting.
3. **Model Serialization:**
   * Exported the trained model binary using `joblib` (`fraud_model.joblib`) for efficient inference in production.

---
