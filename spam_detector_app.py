import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the pre-trained model and vectorizer
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to predict spam or ham
def predict_mail(input_mail):
    input_data_features = vectorizer.transform([input_mail])
    prediction = model.predict(input_data_features)
    return 'Ham mail' if prediction[0] == 1 else 'Spam mail'

# Streamlit app
st.title('Spam Mail Detection App')
st.write('Enter the email text below to predict if it is spam or ham.')

# Text input for email content
input_mail = st.text_area('Email Content', '')

# Predict button
if st.button('Predict'):
    if input_mail:
        result = predict_mail(input_mail)
        st.write(f'The email is classified as: **{result}**')
    else:
        st.write('Please enter some email content to classify.')

# Display instructions
st.sidebar.title("Instructions")
st.sidebar.info(
    """
    Enter the email content in the text box and click on the **Predict** button 
    to see if the email is classified as spam or ham.
    """
)
