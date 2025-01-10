import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tk = pickle.load(open("d:/PROJECTS/AIML  internship/SMS SPAM DETECTION USING NLP/vectorizer.pkl", 'rb'))
model = pickle.load(open("d:/PROJECTS/AIML  internship/SMS SPAM DETECTION USING NLP/model.pkl", 'rb'))

st.title("SMS Spam Detection Model")
st.write("*Made by KUNJ SHAH*")
    

input_sms = st.text_input("Enter the SMS")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tk.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")