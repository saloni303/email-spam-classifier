import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# ✅ Load the correctly saved files
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter a message")

if st.button("Predict"):
    # Step 1: Preprocess
    transformed_sms = transform_text(input_sms)

    # Step 2: Vectorize  ← THIS was line 56, now fixed
    vector_input = tfidf.transform([transformed_sms])

    # Step 3: Predict
    result = model.predict(vector_input)[0]

    # Step 4: Display
    if result == 1:
        st.header("🚨 Spam")
    else:
        st.header("✅ Not Spam")