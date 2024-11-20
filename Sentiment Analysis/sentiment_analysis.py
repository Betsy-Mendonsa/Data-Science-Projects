from joblib import load
import streamlit as st
 
# Load trained model and vectorizer
model = load('best_model.joblib')
vectorizer = load('vectorizer.joblib')  

# Function to predict sentiment
def predict_sentiment(text):
    text_vectorized = vectorizer.transform([text])  
    prediction = model.predict(text_vectorized)[0]
    sentiment = "Positive" if prediction == 0 else "Negative"
    return sentiment

st.title("Sentiment Analysis")
st.subheader("Analyze the sentiment of your sentence or review instantly!")

st.markdown("**Enter your text below to predict whether the sentiment is positive or negative:**")

user_input = st.text_input("Type your sentence here...")
if st.button("Analyze Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter a sentence to analyze.")





  



