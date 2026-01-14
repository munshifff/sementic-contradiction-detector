import streamlit as st
from fake_review_analyzer import SemanticContradictionDetector

# --- Streamlit App ---
st.set_page_config(page_title="Fake Review Analyzer", page_icon="🔍")

st.title("🔍 Fake Review Analyzer")
st.markdown("""
This app detects **semantic contradictions** within a piece of text (like a product review). 
It uses the `SemanticContradictionDetector` class imported from `fake_review_analyzer.py`.
""")

@st.cache_resource
def load_detector():
    # This will initialize the model using the class from your file
    return SemanticContradictionDetector()

with st.spinner("Loading model... This may take a minute on the first run."):
    detector = load_detector()

text_input = st.text_area("Enter the review text to analyze:", height=200, placeholder="Example: The food was great. I hated the meal.")

if st.button("Analyze Text"):
    if text_input.strip():
        with st.spinner("Analyzing..."):
            # Using the analyze method from your class
            result = detector.analyze(text_input)
            
            st.subheader("Results")
            if result.has_contradiction:
                st.error(f"Contradiction Detected! (Confidence: {result.confidence:.2f})")
                st.write(f"**Explanation:** {result.explanation}")
                
                st.write("### Contradicting Pairs:")
                for i, (p1, p2) in enumerate(result.contradicting_pairs):
                    with st.expander(f"Pair {i+1}"):
                        st.write(f"**Sentence 1:** {p1}")
                        st.write(f"**Sentence 2:** {p2}")
            else:
                st.success("No contradictions detected.")
                st.write(f"**Explanation:** {result.explanation}")
    else:
        st.warning("Please enter some text to analyze.")

st.sidebar.header("About")
st.sidebar.info("""
This tool is designed to help identify potentially fake or inconsistent reviews by checking for internal logical contradictions.
""")