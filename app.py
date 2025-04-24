import streamlit as st
import torch
import gdown
import os
import numpy as np
import nltk
from nltk.corpus import wordnet
import re

# Assuming TextClassifier and preprocess function are defined elsewhere

# Download the model from Google Drive if not already downloaded
model_url = "https://drive.google.com/uc?export=download&id=1ifHFDdCIPnC9rpyhnhRy98zy4Ruo3wKR"
save_path = "sentiment_model_state_dict.pth"

if not os.path.exists(save_path):
    gdown.download(model_url, save_path, quiet=False)

# Load the model
# Assuming `YourModelClass` is the model class used when the model was saved
# model = YourModelClass()  # Replace with your actual model class

# Load the model state dict and apply it to the model architecture
model = torch.load(save_path)
model.eval()  # Put the model in evaluation mode

# Define preprocess function for input text
def preprocess(text):
    # Basic text preprocessing
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = ' '.join([word for word in text.split() if word not in nltk.corpus.stopwords.words('english')])
    return text

# Define prediction function (assume your model takes tokenized input)
def predict(text, model):
    # Preprocess the text
    processed_text = preprocess(text)
    
    # Tokenize the text (Assuming you have a function or model that does this)
    # Assuming a function `tokenize` or similar. Replace with the actual tokenizer
    tokenized_input = torch.tensor([processed_text])  # Example: this needs actual tokenization
    
    # Make prediction
    with torch.no_grad():
        output = model(tokenized_input)
    return output

# Streamlit app UI
st.title('Sentiment Analysis')

st.write("This app uses sentiment analysis on stock-related messages from StockTwits.")

# Input text field for user
user_input = st.text_area("Enter a message to analyze sentiment:")

if user_input:
    # Preprocess and make prediction
    pred = predict(user_input, model)
    sentiment = np.argmax(pred.detach().numpy())
    
    sentiments = ["Negative", "Neutral", "Positive"]
    sentiment_label = sentiments[sentiment]
    
    st.write(f"The sentiment for the message is: **{sentiment_label}**")
    
    # Provide further details
    st.write("Prediction Vector: ", pred)
