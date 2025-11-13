import streamlit as st
import pickle
import numpy as np

# ------------------------------------------------------------
# Load Saved Model + Vectorizer
# ------------------------------------------------------------
model = pickle.load(open("intent_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Class labels from Banking77 dataset (77 intents)
banking_labels = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer", "balance_not_updated_after_card_payment",
    "balance_not_updated_after_cash_withdrawal", "beneficiary_not_allowed", "cancel_transfer",
    "card_about_to_expire", "card_acceptance", "card_arrival", "card_delivery_estimate", 
    "card_linking", "card_not_working", "card_payment_fee_charged", "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge", "cash_withdrawal_not_recognised",
    "cash_withdrawal_wrong_exchange_rate", "change_pin", "charged_twice", "check_balance",
    "compromised_card", "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer", "disposable_card", "edit_personal_details",
    "exchange_charge", "exchange_rate", "extra_charge_on_statement", "failed_transfer",
    "freeze_card", "insurance_question", "invalid_card_number", "lost_or_stolen_card",
    "lost_or_stolen_phone", "missing_cash_withdrawal", "money_not_received_by_recipient",
    "money-received", "open_dispute", "passcode_forgotten", "payment_declined",
    "pending_card_payment", "pending_cash_withdrawal", "pending_top_up", "pin_blocked",
    "receiving_money", "refund_not_received", "request_refund", "reverted_card_payment",
    "supported_cards_and_currencies", "terminate_account", "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed", "transfer_fee_charged",
    "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity", "verify_my_identity", "virtual_card_not_working",
    "visa_or_mastercard", "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal"
]

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="QuerySense AI", page_icon="🤖")

st.title("🤖 QuerySense AI — Intent Classification")
st.write("Enter a customer query, and the AI will identify the intent.")

user_input = st.text_area("✍️ Enter customer message:")

if st.button("Predict Intent"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Transform input
        vec = vectorizer.transform([user_input])
        
        # Predict
        pred = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]
        confidence = np.max(probs) * 100

        predicted_label = banking_labels[pred]

        st.success(f"🎯 **Predicted Intent:** {predicted_label}")
        st.info(f"📊 **Confidence:** {confidence:.2f}%")

        st.write("----")
        st.write("### 🔍 Probability Breakdown (Top 5)")
        
        # Show top 5 labels
        top_idx = probs.argsort()[-5:][::-1]
        for idx in top_idx:
            st.write(f"- **{banking_labels[idx]}** → {probs[idx]*100:.2f}%")
