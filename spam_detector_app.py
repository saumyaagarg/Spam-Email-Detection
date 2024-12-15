"""Spam Email Prediction with Streamlit"""

# Importing the Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib
import streamlit as st

# Download required nltk resources
nltk.download('punkt')
nltk.download('stopwords')

# Data Preprocessing
def transform_text(message):
    """Preprocess the text data."""
    ps = PorterStemmer()
    # Convert to lowercase
    message = message.lower()
    # Tokenize the text
    message = nltk.word_tokenize(message)
    # Remove special characters
    message = [word for word in message if word.isalnum()]
    # Remove stopwords and punctuation
    message = [word for word in message if word not in stopwords.words('english') and word not in string.punctuation]
    # Stemming
    message = [ps.stem(word) for word in message]
    return " ".join(message)

# Loading and Preprocessing Data
def load_data():
    """Load the dataset and preprocess it."""
    raw_mail_data = pd.read_csv('mail_data.csv')
    raw_mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')  # Replace NaN values with empty strings
    raw_mail_data['Category'] = raw_mail_data['Category'].apply(lambda x: 0 if x == 'spam' else 1)  # Encode labels
    raw_mail_data['transformed_message'] = raw_mail_data['Message'].apply(transform_text)  # Preprocess text
    return raw_mail_data

# Training the Model
def train_model(data):
    """Train a Logistic Regression model on the preprocessed data."""
    X = data['transformed_message']
    Y = data['Category'].astype(int)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

    # Feature extraction using TfidfVectorizer
    tfidf = TfidfVectorizer(min_df=1, stop_words='english')
    X_train_features = tfidf.fit_transform(X_train)
    X_test_features = tfidf.transform(X_test)

    # Train Logistic Regression
    model = LogisticRegression()
    model.fit(X_train_features, Y_train)

    # Evaluate Model
    train_accuracy = accuracy_score(Y_train, model.predict(X_train_features))
    test_accuracy = accuracy_score(Y_test, model.predict(X_test_features))

    print(f"Training Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")

    # Save the Model and Vectorizer
    joblib.dump(model, 'logistic_regression_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

    return model, tfidf

# Streamlit App
def spam_email_detector():
    """Run the Streamlit app."""
    st.title('Spam Email Detector')
    st.write("Classify emails as Spam or Ham (Not Spam) using a Logistic Regression model.")

    # Load the model and vectorizer
    model = joblib.load('logistic_regression_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')

    # User Input
    user_input = st.text_area('Enter the email text below:')

    # Predict Button
    if st.button('Predict'):
        if user_input.strip():
            # Transform user input using the TfidfVectorizer
            user_input_transformed = tfidf.transform([user_input])
            # Make prediction
            prediction = model.predict(user_input_transformed)
            # Display result
            if prediction[0] == 1:
                st.success('The email is **Ham (Not Spam)**.')
            else:
                st.error('The email is **Spam**.')
        else:
            st.warning('Please enter some email text!')

# Main Function
if __name__ == '__main__':
    # Comment out the next two lines after training the model once
    data = load_data()
    train_model(data)
    
    # Run the Streamlit app
    spam_email_detector()
