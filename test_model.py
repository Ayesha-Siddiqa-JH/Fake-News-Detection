import pickle

# 📌 Load the saved model and vectorizer
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)

    print("\n✅ Model and vectorizer loaded successfully!")

except FileNotFoundError:
    print("\n❌ Model or vectorizer not found! Make sure they are saved correctly.")
    exit()

# 📌 Function to predict fake or real news
def predict_news(news):
    if not news.strip():
        return "⚠️ Please enter valid news content."

    # Transform the news into TF-IDF vector
    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)
    
    # Display the result with formatting
    return "\n🔥 Prediction: 📰 " + ("Fake News 🚫" if prediction[0] == 1 else "Real News ✅")

# 📌 Test with sample news
test_news = input("\n💡 Enter the news content: ")
result = predict_news(test_news)
print(result)

