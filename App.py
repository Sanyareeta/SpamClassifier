import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download stopwords if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    
    # Remove non-alphanumeric characters
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    # Filter out stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()
    
    # Initialize the stemmer
    ps = PorterStemmer()
    
    # Apply stemming
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# tfidf=pickle.load('vectorizer.pkl','rb')
# model=pickle.load('model.pkl','rb')
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
st.title("SMS Spam Classifier")
input_sms=st.text_input("Enter the message")
if st.button("Predict"):
 #1.preprocess
 transformed_sms=transform_text(input_sms)
#2.vectorize
 vector_input=tfidf.transform([transformed_sms])
#3.predict
 result=model.predict(vector_input)[0]
#4.Display 
 if result==1:
    st.header("Spam")
 else:
    st.header("Not spam")