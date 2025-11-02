# ===============================
# 📧 EMAIL SPAM DETECTION APP
# Built with Streamlit + Naive Bayes
# ===============================

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# ⚙️ PAGE CONFIGURATION
# ===============================
st.set_page_config(page_title="Email Spam Detection", layout="wide")

st.title("📧 Email Spam Detection System")
st.markdown("### Detect whether an email message is **Spam** or **Ham** using Machine Learning (Naive Bayes).")

# ===============================
# 📂 LOAD DATASET
# ===============================
uploaded_file = st.file_uploader("📤 Upload your CSV dataset (like newemail.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')

    # Clean Dataset
    df.rename(columns={"v1": "Category", "v2": "Message"}, inplace=True)
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True, errors='ignore')
    df['Spam'] = df['Category'].astype(str).apply(lambda x: 1 if x == 'spam' else 0)

    st.success("✅ Dataset successfully loaded!")
    st.write("**Dataset Preview:**")
    st.dataframe(df.head())

    # ===============================
    # 📊 DATA ANALYSIS
    # ===============================
    st.subheader("📊 Data Insights")

    col1, col2 = st.columns(2)

    with col1:
        spread = df['Category'].value_counts()
        fig, ax = plt.subplots()
        spread.plot(kind='pie', autopct='%1.1f%%', cmap='Set2', ax=ax)
        ax.set_ylabel('')
        ax.set_title("Distribution of Spam vs Ham")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        spread.plot(kind='bar', color=['#ff9999', '#66b3ff'], ax=ax)
        for index, value in enumerate(spread):
            ax.text(index, value + 10, str(value), ha='center')
        ax.set_title("Count of Spam vs Ham")
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # ===============================
    # ☁️ WORDCLOUD
    # ===============================
    st.subheader("☁️ Most Used Words in Spam Messages")
    df_spam = df[df['Category'] == 'spam']

    comment_words = ''
    stopwords = set(STOPWORDS)
    for val in df_spam.Message:
        val = str(val)
        tokens = val.split()
        tokens = [t.lower() for t in tokens]
        comment_words += " ".join(tokens) + " "

    wordcloud = WordCloud(width=1000, height=500,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=10,
                          max_words=1000,
                          colormap='inferno').generate(comment_words)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(wordcloud)
    ax.axis("off")
    st.pyplot(fig)

    # ===============================
    # 🤖 MODEL TRAINING
    # ===============================
    st.subheader("🤖 Model Training")

    X_train, X_test, y_train, y_test = train_test_split(df.Message, df.Spam, test_size=0.25, random_state=42)

    clf = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('nb', MultinomialNB())
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"✅ **Model trained successfully!**")
    st.write(f"📈 **Accuracy:** {accuracy:.2%}")
    st.text(classification_report(y_test, y_pred))

    # Save model for reuse
    joblib.dump(clf, "spam_model.pkl")

    # ===============================
    # 📨 SPAM DETECTION SECTION
    # ===============================
    st.subheader("📨 Try Out Spam Detection")

    email_text = st.text_area("Enter an email message below:")
    if st.button("Detect Spam"):
        prediction = clf.predict([email_text])[0]
        if prediction == 0:
            st.success("✅ This is a **Ham (Safe)** Email!")
        else:
            st.error("🚨 This is a **Spam** Email!")

else:
    st.info("👆 Upload a dataset (like `newemail.csv`) to start training and detection.")
