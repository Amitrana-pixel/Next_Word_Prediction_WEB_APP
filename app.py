import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="AI Next Word Predictor", page_icon="✨", layout="centered")

@st.cache_resource
def load_assets():
    model = load_model("lstm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len

model, tokenizer, max_len = load_assets()

# -------------------- STYLES --------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    font-family: 'Poppins', sans-serif;
}

/* Top Box */
.header-box {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}

.title {
    font-size: 42px;
    font-weight: 700;
    background: linear-gradient(90deg, #00dbde, #fc00ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.tagline {
    font-size: 16px;
    color: #00e6e6;
    margin-top: 5px;
}

.normal-text {
    text-align: center;
    color: #dddddd;
    margin-bottom: 20px;
}

.stTextInput>div>div>input {
    border-radius: 12px;
    padding: 14px;
    font-size: 16px;
}

.stButton>button {
    background: linear-gradient(90deg, #ff8a00, #e52e71);
    color: white;
    border-radius: 30px;
    height: 3.2em;
    font-size: 18px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.08);
    box-shadow: 0 0 15px rgba(255,105,180,0.6);
}

.prediction {
    margin-top: 25px;
    padding: 18px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
}

.footer {
    text-align: center;
    margin-top: 40px;
    font-size: 16px;
}

.name-highlight {
    color: #ff4b2b;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<div class="header-box">
    <div class="title">✨ Next Word Predictor</div>
    <div class="tagline">Predict the next word with intelligence, speed, and style.</div>
</div>
""", unsafe_allow_html=True)

# -------------------- DESCRIPTION --------------------
st.markdown('<div class="normal-text">Smarter text. Faster thoughts. Predict the future — one word at a time.</div>', unsafe_allow_html=True)

# -------------------- INPUT --------------------
user_input = st.text_input("", placeholder="Type your sentence here...")

# -------------------- PREDICTION --------------------
def predict_next_word(text):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""

# -------------------- BUTTON --------------------
if st.button("🚀 Predict Next Word"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first")
    else:
        with st.spinner("Thinking..."):
            word = predict_next_word(user_input)
            st.markdown(f'<div class="prediction">Predicted Word: <b>{word}</b></div>', unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown("""
<div class="footer">
Made with ❤️ by <span class="name-highlight">Amit Kumar Rana</span>
</div>
""", unsafe_allow_html=True)
