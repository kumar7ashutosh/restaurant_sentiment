import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

with open("count_v_res.pkl", 'rb') as cv_file: 
        cv = pickle.load(cv_file)
with open('model.pkl', 'rb') as model_file: 
        model = pickle.load(model_file)
        
custom_stopwords = {'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't",
                    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
                    'needn', "needn't", 'shan', "shan't", 'no', 'nor', 'not', 'shouldn', "shouldn't",
                    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}
all_stopwords = set(stopwords.words("english")) - custom_stopwords
ps = PorterStemmer()

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text) 
    text = text.lower() 
    text = text.split() 
    text = [ps.stem(word) for word in text if word not in all_stopwords] 
    return " ".join(text) 

st.title("Restaurant Review Sentiment Classifier")
st.write("Enter a review to predict if it's positive or negative.")

user_review = st.text_area("Your Review:")

if st.button("Predict Sentiment"):
    if user_review:
        processed_review = preprocess_text(user_review)
        vectorized_review = cv.transform([processed_review]).toarray()
        prediction = model.predict(vectorized_review)[0] 

        if prediction == 1:
            st.success("Positive Review! 👍")
        else:
            st.error("Negative Review! 👎")
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---")
st.caption("Simple Restaurant Review Classifier")