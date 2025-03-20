# 📌 Import Libraries
import pandas as pd
import numpy as np
import re
import string
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords

# 📌 Download NLTK stopwords
nltk.download('stopwords')

# 📌 Load the dataset
true_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')

# Add labels
true_df['label'] = 0  # Real
fake_df['label'] = 1  # Fake

# Combine datasets
df = pd.concat([true_df, fake_df], ignore_index=True)

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# 📌 Text Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)

    # Remove stopwords
    stop_words = stopwords.words('english')
    text = " ".join(word for word in text.split() if word not in stop_words)
    
    return text

# Apply preprocessing
df['text'] = df['title'] + " " + df['text']
df['text'] = df['text'].apply(clean_text)

# 📌 Splitting Data
X = df['text']
y = df['label']

# Vectorizing the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X = vectorizer.fit_transform(X)

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Model with Hyperparameter Tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear']
}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
model = grid_search.best_estimator_

# 📌 Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.2f}")

# 📌 Confusion Matrix and Classification Report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 📌 Save the improved Model and Vectorizer
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("\n✅ Model saved successfully!")

with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)
print("\n✅ Vectorizer saved successfully!")
