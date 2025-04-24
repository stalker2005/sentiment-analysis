import streamlit as st
import torch
import numpy as np
import nltk
from nltk.corpus import wordnet
import re
import os

# Assuming the TextClassifier and preprocess function are already defined.

# Define the path where the model is saved
save_path = r"C:\Users\Asus\OneDrive\Desktop\sentiment-analysis-neural-networks-master\Model"
model_path = os.path.join(save_path, "sentiment_model.pt")

# Load the model
model = torch.load(model_path)
model.eval()

# Streamlit app
st.title('Sentiment Analysis')

st.write("This app uses sentiment analysis on stock-related messages from StockTwits.")

# Input text field for user
user_input = st.text_area("Enter a message to analyze sentiment:")

if user_input:
    # Preprocess and make prediction
    pred = predict(user_input, model, vocab)
    sentiment = np.argmax(pred.detach().numpy())
    
    sentiments = ["Negative", "Neutral", "Positive"]
    sentiment_label = sentiments[sentiment]
    
    st.write(f"The sentiment for the message is: **{sentiment_label}**")
    
    # Provide further details
    st.write("Prediction Vector: ", pred)

