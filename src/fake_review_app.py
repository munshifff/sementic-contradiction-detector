import streamlit as st
from fake_review_analyzer import analyze_all_reviews, evaluate_reviews

st.title("Fake Review Detector")

review_text = st.text_area("Enter a review:")

if st.button("Analyze Review"):
    if review_text.strip():
        reviews = [{"id": 1, "text": review_text}]
        results = analyze_all_reviews(reviews)
        res = results[0]
        st.subheader("Analysis Result")
        st.write(f"Label: {res['label']}")
        st.write(f"Confidence: {res['confidence']}")
        st.write(f"Evidence: {res['evidence']}")
        if res['contradictions']:
            st.write("Contradictions:", res['contradictions'])
        if res['manipulation_patterns']:
            st.write("Manipulation patterns:", res['manipulation_patterns'])
        st.write("Same-author flag:", res['same_author_flag'])
