import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pre-trained model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model('models/hamlet_lstm_model.keras')
    with open('models/hamlet_tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer
model, tokenizer = load_model_and_tokenizer()

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):] # Ensure the sequence length matches the max_sequence_len -1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit app
st.title("Hamlet Next Word Predictor")
input_text = st.text_input("Enter a line from Hamlet and the model will predict the next word:", "")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"The predicted next word is: **{next_word}**" if next_word else "No prediction could be made.")
# Add a footer
st.markdown("---")
