# app.py
import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your trained model and vectorizer
model = joblib.load('logistic_regression_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Streamlit app
st.title('Spam Email Detector')
st.write("Classify emails as Spam or Ham (Not Spam).")

# User input
user_input = st.text_area('Enter the email text below:')

if st.button('Predict'):
    if user_input.strip():  # Ensure input is not empty
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
