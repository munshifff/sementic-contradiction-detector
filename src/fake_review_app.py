import streamlit as st
from fake_review_analyzer import analyze_all_reviews, evaluate_reviews

st.set_page_config(page_title="Fake Review Detector", layout="wide")
st.title("🕵️‍♂️ Fake Review Analyzer")

st.markdown("""
This tool detects:

- Semantic contradictions  
- Psychological manipulation patterns  
- Stylometric fingerprint similarity  

It also supports evaluation if you provide ground-truth labels.
""")

# --------- INPUT REVIEWS ------------
reviews_input = st.text_area(
    "Enter reviews (one per line):",
    placeholder="Paste reviews here...\nOne review per line."
)

# --------- OPTIONAL LABELS ------------
labels_input = st.text_area(
    "Enter ground-truth deceptive labels (True/False, optional, one per line):",
    placeholder="True\nFalse\nTrue"
)

if st.button("Analyze"):
    if not reviews_input.strip():
        st.warning("Please enter at least one review.")
    else:

        # Build review objects expected by your analyzer
        reviews = []
        for i, line in enumerate(reviews_input.strip().split("\n")):
            if not line.strip():
                continue
            reviews.append({
                "id": i + 1,
                "text": line.strip(),
                # optional contradiction span support
                "has_contradiction": False,
                "contradiction_spans": []
            })

        # Parse labels if provided
        ground_truth = None
        if labels_input.strip():
            lbls = [x.strip().lower() for x in labels_input.split("\n") if x.strip()]
            if len(lbls) != len(reviews):
                st.error("Number of labels must equal number of reviews.")
            else:
                ground_truth = [
                    {"id": i + 1, "deceptive": (x == "true")}
                    for i, x in enumerate(lbls)
                ]

        st.info("Running detection pipeline…")
        results = analyze_all_reviews(reviews)

        # ----------- SHOW RESULTS -------------
        st.subheader("Results")

        for r in results:
            st.markdown("---")
            st.write(f"**Review ID:** {r['id']}")
            st.write(f"**Deceptive:** {r['deceptive']}")
            st.write(f"**Confidence:** {r['confidence']:.2f}")

            # evidence may be None
            ev = r.get("evidence", None)
            if ev is None:
                st.write("**Evidence:** None")
            else:
                if isinstance(ev, list):
                    st.write("**Evidence:** " + ", ".join(ev))
                else:
                    st.write(f"**Evidence:** {ev}")

        # ----------- EVALUATION -------------
        if ground_truth is not None:
            st.subheader("Evaluation Metrics")

            metrics = evaluate_reviews(results, ground_truth)

            st.write(f"Accuracy: **{metrics['accuracy']:.2f}**")
            st.write(f"Precision: **{metrics['precision']:.2f}**")
            st.write(f"Recall: **{metrics['recall']:.2f}**")
            st.write(f"F1 Score: **{metrics['f1']:.2f}**")
