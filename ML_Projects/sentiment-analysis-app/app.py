import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("best_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page Config
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="centered"
)

# Sidebar
st.sidebar.title("📊 Sentiment Categories")

st.sidebar.success("😊 Positive")
st.sidebar.info("😐 Neutral")
st.sidebar.error("😞 Negative")

st.sidebar.markdown("---")
st.sidebar.write("""
**Positive**: Happy, satisfied, favorable reviews.

**Neutral**: Neither positive nor negative.

**Negative**: Unhappy, dissatisfied, unfavorable reviews.
""")

# Main Title
st.title("💬 Sentiment Analysis App")

review = st.text_area(
    "Enter your review",
    height=150
)

if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review")

    else:

        review_vector = vectorizer.transform([review])

        prediction = model.predict(review_vector)[0]

        probs = model.predict_proba(review_vector)[0]
        classes = model.classes_

        # Create probability dictionary
        prob_dict = {
            str(label).lower(): prob * 100
            for label, prob in zip(classes, probs)
        }

        positive_score = prob_dict.get("positive", 0)
        neutral_score = prob_dict.get("neutral", 0)
        negative_score = prob_dict.get("negative", 0)

        # Predicted Sentiment
        if str(prediction).lower() == "positive":
            st.success("😊 Positive Sentiment")

        elif str(prediction).lower() == "neutral":
            st.info("😐 Neutral Sentiment")

        else:
            st.error("😞 Negative Sentiment")

        st.markdown("### Confidence Scores")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("😊 Positive", f"{positive_score:.2f}%")

        with col2:
            st.metric("😐 Neutral", f"{neutral_score:.2f}%")

        with col3:
            st.metric("😞 Negative", f"{negative_score:.2f}%")

        # Progress Bars
        st.markdown("### Sentiment Distribution")

        st.write("Positive")
        st.progress(min(int(positive_score), 100))

        st.write("Neutral")
        st.progress(min(int(neutral_score), 100))

        st.write("Negative")
        st.progress(min(int(negative_score), 100))