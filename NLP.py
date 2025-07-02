import streamlit as st
import joblib
import pandas as pd

# Load models
spam_model = joblib.load("spam.pkl")
lan_model = joblib.load("lang_det.pkl")
news_model = joblib.load("news_short.pkl")
review_model = joblib.load("review.pkl")

# Page config
st.set_page_config(layout="wide", page_title="LENS eXpert", page_icon="🔍")

# Custom background and style
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #f3f4f7 0%, #dfe9f3 100%);
        }
        .stTextInput > div > div > input {
            font-size: 20px !important;
            color: #333333;
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown("""
    <h1 style='background-color: #90EE90; 
                font-size: 36px; 
                color: #000; 
                padding: 10px; 
                border-radius: 12px; 
                text-align: center;
                margin-top: -30px;'>
         🔍 LENS eXpert : The NLP Toolkit for Smart Text Analysis
    </h1>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📨 Spam Classifier", "🌐 Language Detection", "🍽️ Food Review Sentiment", "🗞️ News Classification"])

# --- Tab 1: Spam Classifier ---
with tab1:
    st.markdown("<h2 style='color: #DC143C;'>📨 Welcome to the Spam Classifier</h2>", unsafe_allow_html=True)
    msg1 = st.text_input("📩 Paste or type your email/message to check for spam:", key="msg_input1")
    if st.button("🔍 Predict", key="spam_detection"):
        pred = spam_model.predict([msg1])
        if pred[0] == 0:
            st.image("spam.jpg", caption="Spam Detected ❌")
        else:
            st.image("not_spam.png", caption="Not Spam ✅")

    uploaded_file = st.file_uploader("📁 Upload a file (CSV/TXT):", type=["csv", "txt"], key="spam_file")
    if uploaded_file:
        df_spam = pd.read_csv(uploaded_file, header=None, names=["Msg"])
        pred = spam_model.predict(df_spam.Msg)
        df_spam.index = range(1, df_spam.shape[0] + 1)
        df_spam["Prediction"] = pred
        df_spam["Prediction"] = df_spam["Prediction"].map({0: "❌ Spam", 1: "✅ Not Spam"})
        st.dataframe(df_spam)

# --- Tab 2: Language Detection ---
with tab2:
    st.markdown("<h2 style='color: #00008B;'>🌐 Welcome to Language Detection</h2>", unsafe_allow_html=True)
    
    msg2 = st.text_input("📝 Enter a sentence or paragraph to detect the language:", key="msg_input2")

    if st.button("🔍 Predict", key="language_detection"):
        pred = lan_model.predict([msg2])
        st.success(f"Detected Language: {pred[0]}")

    uploaded_file = st.file_uploader("📁 Upload a file (CSV/TXT):", type=["csv", "txt"], key="lan_file")

    if uploaded_file:
        df_lan = pd.read_csv(uploaded_file, header=None, names=["Msg"])
        pred = lan_model.predict(df_lan.Msg)
        df_lan.index = range(1, df_lan.shape[0] + 1)
        df_lan["Prediction"] = pred
        st.dataframe(df_lan)

# --- Tab 3: Food Review Sentiment ---
with tab3:
    st.markdown("<h2 style='color: #FF8C00;'>🍽️ Welcome to Food Review Sentiments</h2>", unsafe_allow_html=True)
    msg3 = st.text_input("🍔 Share a food review and we'll detect if it's positive or negative:", key="msg_input3")
    if st.button("🔍 Predict", key="review_detection"):
        pred = review_model.predict([msg3])
        if pred[0] == 1:
            st.image("liked.jpeg", caption="👍 Liked")
        else:
            st.image("images.jpeg", caption="👎 Disliked")

    uploaded_file = st.file_uploader("📁 Upload a file (CSV/TXT):", type=["csv", "txt"], key="review_file")
    if uploaded_file:
        df_review = pd.read_csv(uploaded_file, header=None, names=["Msg"])
        pred = review_model.predict(df_review.Msg)
        df_review.index = range(1, df_review.shape[0] + 1)
        df_review["Prediction"] = pred
        df_review["Prediction"] = df_review["Prediction"].map({0: "👎 Disliked", 1: "👍 Liked"})
        st.dataframe(df_review)

# --- Tab 4: News Classification ---
with tab4:
    st.markdown("<h2 style='color: #006400;'>🗞️ Welcome to News Classification</h2>", unsafe_allow_html=True)
    
    msg4 = st.text_input("📰 Enter a news headline or article to classify the topic:", key="msg_input4")
    
    if st.button("🔍 Predict", key="news_detection"):
        pred = news_model.predict([msg4])
        st.success(f"📰 {pred[0]}")

    uploaded_file = st.file_uploader("📁 Upload a file (CSV/TXT):", type=["csv", "txt"], key="news_file")
    if uploaded_file:
        df_news = pd.read_csv(uploaded_file, header=None, names=["Msg"])
        pred = news_model.predict(df_news.Msg)
        df_news.index = range(1, df_news.shape[0] + 1)
        df_news["Prediction"] = pred
        st.dataframe(df_news)

# --- Sidebar ---

st.sidebar.image("riteshsamridhipics.jpg")


with st.sidebar.expander("🌐 About Us"):
    st.markdown("""
    <div style='font-size: 15px;'>
        We are final-year students passionate about <b>Data Science</b> & <b>Natural Language Processing</b>.<br><br>
        This app is part of our academic project to help users test multiple NLP models with ease.
    </div>
    """, unsafe_allow_html=True)

with st.sidebar.expander("📞 Contact us"):
    st.write("📱 7061931957")
    st.write("✉️ 07nk05@gmail.com")

with st.sidebar.expander("🤝 Help & Instructions"):
    st.markdown("""
    <ul style='font-size: 14px;'>
        <li>Type or upload text to test the model.</li>
        <li>Use supported file formats: <b>.csv</b> or <b>.txt</b>.</li>
        <li>After prediction, download the result using the button.</li>
    </ul>
    """, unsafe_allow_html=True)

st.markdown("""
    <hr style="margin-top: 40px; border: none; border-top: 2px solid #ccc;" />
    <div style="text-align: center; padding: 10px; font-size: 24px; color: #555;">
        🔍 <b>LENS eXpert</b> | Built with ❤️ using <b>Streamlit</b>, <b>sklearn</b>, and <b>nltk</b><br>
        📚 For Learning & Academic Use | © 2025
    </div>
""", unsafe_allow_html=True)