# fake_review_app.py

import streamlit as st
from fake_review_analyzer import analyze_all_reviews, evaluate_reviews  # Make sure evaluate_reviews exists

st.set_page_config(page_title="Fake Review Detector", layout="wide")
st.title("🕵️‍♂️ Fake Review Analyzer")

st.markdown("""
Enter one or more reviews below. The system will detect:
- Semantic contradictions
- Psychological manipulation
- Stylometric similarity (same-author fingerprint)

You can also provide ground truth labels to evaluate performance if available.
""")

# --- Input reviews ---
reviews_input = st.text_area(
    "Enter reviews (one per line):",
    placeholder="Write/paste reviews here, each on a new line..."
)

# --- Optional ground truth labels ---
labels_input = st.text_area(
    "Enter corresponding ground truth labels (True/False, one per line, optional):",
    placeholder="True\nFalse\nTrue\n..."
)

if st.button("Analyze Reviews"):
    if not reviews_input.strip():
        st.warning("Please enter at least one review to analyze.")
    else:
        # --- Prepare reviews list ---
        reviews = [
            {"id": idx + 1, "text": line.strip()}
            for idx, line in enumerate(reviews_input.strip().split("\n"))
            if line.strip()
        ]

        # --- Prepare labels if provided ---
        labels = None
        if labels_input.strip():
            labels_lines = [line.strip().lower() for line in labels_input.strip().split("\n") if line.strip()]
            if len(labels_lines) != len(reviews):
                st.error("Number of labels must match number of reviews!")
            else:
                labels = [line == "true" for line in labels_lines]

        st.info(f"Analyzing {len(reviews)} review(s)...")

        # --- Run analysis ---
        results = analyze_all_reviews(reviews)  # Make sure this returns 'text' in each dict

        # --- Display results ---
        st.subheader("Analysis Results")
        for res in results:
            st.markdown("---")
            st.subheader(f"Review ID: {res['id']}")
            st.write(f"**Text:** {res['text']}")
            st.write(f"**Deceptive:** {res['deceptive']}")
            st.write(f"**Confidence:** {res['confidence']:.2f}")
            st.write(f"**Evidence:** {', '.join(res['evidence']) if res['evidence'] else 'None'}")

        # --- Evaluate if labels provided ---
        if labels:
            st.subheader("Evaluation Metrics")
            metrics = evaluate_reviews(results, labels)  # Make sure this function exists
            st.write(f"**Accuracy:** {metrics['accuracy']:.2f}")
            st.write(f"**Precision:** {metrics['precision']:.2f}")
            st.write(f"**Recall:** {metrics['recall']:.2f}")
            st.write(f"**F1 Score:** {metrics['f1']:.2f}")
