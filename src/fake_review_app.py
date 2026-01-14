# fake_review_app.py

import streamlit as st
from typing import List, Dict
from fake_review_analyzer import SemanticContradictionDetector, evaluate, ContradictionResult

st.set_page_config(page_title="Semantic Contradiction Detector", layout="wide")
st.title("🕵️‍♂️ Semantic Contradiction Detector")

st.markdown("""
Enter one or more reviews below. The system will detect semantic contradictions in each review.
You can also provide ground truth labels (`True`/`False`) to evaluate the performance.
""")

# ----------------------- INPUT -----------------------
reviews_input = st.text_area(
    "Enter reviews (one per line):",
    placeholder="Write/paste reviews here, each on a new line..."
)

labels_input = st.text_area(
    "Enter corresponding ground truth labels (True/False, one per line, optional):",
    placeholder="True\nFalse\nTrue\n..."
)

if st.button("Analyze Reviews"):

    if not reviews_input.strip():
        st.warning("Please enter at least one review to analyze.")
    else:
        # Prepare reviews list
        reviews = [
            {"id": idx + 1, "text": line.strip()}
            for idx, line in enumerate(reviews_input.strip().split("\n"))
            if line.strip()
        ]

        # Prepare labels if provided
        labels: List[bool] = None
        if labels_input.strip():
            label_lines = [line.strip().lower() for line in labels_input.strip().split("\n") if line.strip()]
            if len(label_lines) != len(reviews):
                st.error("Number of labels must match number of reviews!")
            else:
                labels = [line == "true" for line in label_lines]

        st.info(f"Analyzing {len(reviews)} review(s)... This may take a few seconds.")

        # ----------------------- RUN ANALYSIS -----------------------
        detector = SemanticContradictionDetector()
        results: List[ContradictionResult] = []
        for r in reviews:
            res = detector.analyze(r["text"])
            results.append(res)

        # ----------------------- DISPLAY RESULTS -----------------------
        st.subheader("Analysis Results")
        for idx, r in enumerate(reviews):
            res = results[idx]
            st.markdown("---")
            st.subheader(f"Review ID: {r['id']}")
            st.write(f"**Text:** {r['text']}")
            st.write(f"**Has Contradiction:** {res.has_contradiction}")
            st.write(f"**Confidence:** {res.confidence:.2f}")
            st.write(f"**Contradicting Pairs:** {res.contradicting_pairs if res.contradicting_pairs else 'None'}")
            st.write(f"**Explanation:** {res.explanation}")

        # ----------------------- EVALUATION -----------------------
        if labels:
            st.subheader("Evaluation Metrics")
            # Add 'has_contradiction' key to match evaluate function
            for idx, r in enumerate(reviews):
                r["has_contradiction"] = labels[idx]
            metrics = evaluate(detector, reviews)
            st.write(f"**Accuracy:** {metrics['accuracy']:.2f}")
            st.write(f"**Precision:** {metrics['precision']:.2f}")
            st.write(f"**Recall:** {metrics['recall']:.2f}")
            st.write(f"**F1 Score:** {metrics['f1']:.2f}")
