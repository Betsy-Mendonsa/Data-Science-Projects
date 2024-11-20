import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the encoder, scaler, and trained model
encoder = joblib.load('label_encoder.joblib')  
scaler = joblib.load('scalar.joblib')         
dt_model = joblib.load('dt_model.joblib')    

# Load dataset
df = pd.read_csv('updated_fraud_data.csv')

# Define categorical and numerical columns
categorical_columns = ['merchant', 'category', 'job', 'location']
numerical_columns = ['amt', 'trans_hour', 'age']
training_order = ['merchant', 'category', 'amt', 'job', 'trans_hour', 'age', 'location']  

# Extract unique values for dropdowns
merchants = df['merchant'].unique().tolist()
categories = df['category'].unique().tolist()
jobs = df['job'].unique().tolist()
locations = df['location'].unique().tolist()

# Streamlit app title
st.title("Fraud Detection System")

# Input form
st.sidebar.header("Input New Transaction Details")
merchant = st.sidebar.selectbox("Merchant", merchants)
category = st.sidebar.selectbox("Category", categories)
amt = st.sidebar.number_input("Transaction Amount", min_value=1, max_value=10000, value=80)
job = st.sidebar.selectbox("Job", jobs)
trans_hour = st.sidebar.slider("Transaction Hour", min_value=0, max_value=23, value=10)
age = st.sidebar.number_input("Customer Age", min_value=18, max_value=100, value=25)
location = st.sidebar.selectbox("Location", locations)

# Create a DataFrame for the input
x_test_unseen = pd.DataFrame(
    [[merchant, category, amt, job, trans_hour, age, location]],
    columns=training_order
)

# Handle unseen categories and encode categorical columns
for col in categorical_columns:
    x_test_unseen[col] = x_test_unseen[col].apply(
        lambda val: val if val in encoder.classes_ else 'unknown'
    )
    if 'unknown' not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, 'unknown')
    x_test_unseen[col] = encoder.transform(x_test_unseen[col])

# Scale all columns
x_test_unseen_scaled = scaler.transform(x_test_unseen)

# Predict using the trained model
if st.sidebar.button("Predict"):
    prediction = dt_model.predict(x_test_unseen_scaled)
    prediction_text = "Fraudulent" if prediction[0] == 1 else "Non-Fraudulent"
    st.subheader("Prediction Result")
    st.write(f"The transaction is predicted to be ***{prediction_text}***.")






