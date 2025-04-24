import streamlit as st
import torch
import gdown
import os
import numpy as np
import nltk
from nltk.corpus import wordnet
import re

# Assuming the TextClassifier and preprocess function are already defined.

# Define the URL for the Google Drive model file
model_url = "https://drive.google.com/uc?export=download&id=1ifHFDdCIPnC9rpyhnhRy98zy4Ruo3wKR"
save_path = "sentiment_model.pt"

# Download the model from Google Drive if not already downloaded
if not os.path.exists(save_path):
    gdown.download(model_url, save_path, quiet=False)

# Load the model
model = torch.load(save_path)
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
