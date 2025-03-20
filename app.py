import streamlit as st
import pickle
import pandas as pd

# Load the model and vectorizer
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Function to predict fake or real news
def predict_news(news):
    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)
    return "🛑 Fake News" if prediction[0] == 1 else "✅ Real News"

# Streamlit UI
st.title("📰 Fake News Detection System")
st.write("🔍 Enter the news content below to detect if it's **real or fake**.")

# Text input
news = st.text_area("Enter News Content:", height=200)

# Prediction button
if st.button("Predict"):
    if news.strip():
        result = predict_news(news)
        st.success(f"**Prediction:** {result}")
    else:
        st.warning("⚠️ Please enter some news content to predict!")

# Footer
st.markdown("---")
st.write("💙 Created by **Ayesha Siddiqa JH** 🚀")

