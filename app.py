import streamlit as st
import pickle

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
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




# Streamlit UI
st.title("📩 SMS Spam Detection with Sender Lookup")

sms_text = st.text_area("Enter the SMS text:")

if st.button("Classify SMS"):
    if sms_text.strip() != "":
        processed_text = transform_text(sms_text)
        vectorized_text = vectorizer.transform([processed_text])

        # Prediction & probability
        prediction = model.predict(vectorized_text)[0]
        probability = model.predict_proba(vectorized_text)[0][1]  # spam probability

        if prediction == 1:  # spam
            st.warning("⚠️ This SMS is classified as **Spam**.")

            # Show confidence score
            st.subheader("📊 Model Confidence")
            st.progress(int(probability * 100))
            st.write(f"Spam probability: **{probability * 100:.2f}%**")

            # Highlight spam keywords
            spam_keywords = ["free", "win", "click", "prize", "claim", "urgent", "lottery", "offer", "congratulations"]
            highlighted = sms_text
            for word in spam_keywords:
                highlighted = re.sub(
                    rf"\b({word})\b",
                    r"<mark>\1</mark>",
                    highlighted,
                    flags=re.IGNORECASE
                )
            st.subheader("🔍 Highlighted Spam Keywords")
            st.markdown(highlighted, unsafe_allow_html=True)


        else:
            st.success("✅ This SMS is classified as **Not Spam**.")
            st.write(f"Ham probability: **{(1 - probability) * 100:.2f}%**")
    else:
        st.error("Please enter an SMS text to classify.")
