import streamlit as st
import pickle

# Load saved model + vectorizer
model = pickle.load(open("intent_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Banking77 label list
from datasets import load_dataset
label_list = load_dataset("banking77")["train"].features["label"].names

st.set_page_config(page_title="QuerySense AI", page_icon="🔍")
st.title("🔍 QuerySense AI — Intent Classification")
st.write("Enter your query below and the model will identify the user's **intent**.")

text = st.text_area("User Query:", height=120)

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter a query.")
    else:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec).max() * 100

        st.success(f"### 🧠 Predicted Intent: **{label_list[pred]}**")
        st.write(f"📊 **Confidence:** `{proba:.2f}%`")
