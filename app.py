import streamlit as st
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "spam_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))


# Set up the Streamlit app

st.set_page_config(page_title = "Spam Classifier",layout = "centered")


st.title("📩 Spam vs Non-Spam Classifier")
st.write("Enter a message below to check whether it is spam or not.")

# Input text area for user message
user_input = st.text_area("Enter message text")


if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify.")

    else:
        text_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(text_vectorized)[0]
        prob = model.predict_proba(text_vectorized)[0][1]
           
        if prediction == 1:
            st.error("🚫 The message is classified as SPAM.")
            st.write(f"Spam probability: {prob:.2f}")

        else:
            st.success("✅ The message is classified as NON-SPAM.")    
