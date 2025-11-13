# 🔍 QuerySense AI — Intent Classification for User Queries

QuerySense AI is a machine-learning powered system that **understands the intent** behind user queries.  
It classifies text into **77 banking-related intents** using a trained ML model.

This project includes:

- 🧠 A **trained ML model** (TF-IDF + Logistic Regression)  
- ⚙️ A **Streamlit web app** for real-time predictions  
- 📘 Google Colab training notebook  
- 📝 Sample inputs & outputs  
- 🚀 Lightweight, fast, and runs offline  

---

## ✨ Features

- ✔️ Classifies queries into 77 intent categories  
- ✔️ Fast training (~few seconds on Colab)  
- ✔️ Uses **Banking77** dataset  
- ✔️ Clean, professional UI  
- ✔️ Confidence score displayed  
- ✔️ Saved model + vectorizer included  

---

## 🧠 Intents Recognized

Example intents from the Banking77 dataset:

- “card_arrival”
- “apple_pay_not_working”
- “transfer_timing”
- “balance_not_updated”
- “disposable_card_limits”
- “pending_card_payment”
- …and **71 more**

Model predicts **one of 77 intents** for any input query.

---

## 🧪 Sample Inputs & Outputs

### Input
When will my new card arrive?

### Output
card_arrival
