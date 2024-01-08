import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
import pickle
import streamlit as st
import re

#load the data
df = pd.read_csv("E:\my apps\Phishing_Email.csv")
new_df = df.drop(['Unnamed: 0'],axis=1)
new_df = new_df.dropna(axis=0)
new_df = new_df.drop_duplicates()

#label encoding
lbl = LabelEncoder()
new_df['Email Type'] = lbl.fit_transform(new_df['Email Type'])


def preprocess_text(text):
    # Remove hyperlinks
    text = re.sub(r'http\S+', '', text)

    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Apply the preprocess_text function to the specified column in the DataFrame
new_df["Email Text"] = new_df["Email Text"].apply(preprocess_text)

tf = TfidfVectorizer(stop_words='english',max_features=10000) #dimension reduction
feature_x = tf.fit_transform(new_df['Email Text']).toarray()
y_tf = np.array(new_df['Email Type'])

X_tr,X_tst,y_tr,y_tst = train_test_split(feature_x,y_tf,test_size=0.2,random_state=0)


xgb = MultinomialNB()
xgb.fit(X_tr,y_tr)

def predict_text(input_text):
    process = preprocess_text(input_text)
    vectorized = tf.transform([process])
    prediction = xgb.predict(vectorized[0])
    return "spam" if prediction==1 else "not spam"

def main():
    st.title("Spam Detection")
    st.write("Enter your mail here:")
    user = st.text_area("Enter your message","")
    if st.button("predict"):
        predict = predict_text(user)
        st.write(f"The mail is {predict}")

if __name__=="__main__":
    main()
