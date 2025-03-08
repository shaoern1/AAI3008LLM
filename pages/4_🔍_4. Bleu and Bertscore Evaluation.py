import streamlit as st
from evaluation import compute_bleu, compute_bert_score  # Import evaluation functions

st.set_page_config(
    page_title="Evaluate Model Response",
    page_icon="📝",
)

st.markdown("## Evaluate Model Response")

# Input fields for reference and generated answers
reference = st.text_area("Reference Answer", "")
candidate = st.text_area("Generated Answer", "")

if st.button("Evaluate"):
    if reference and candidate:
        # Compute BLEU and BERTScore
        bleu = compute_bleu(reference, candidate)
        bert = compute_bert_score(reference, candidate)

        # Display BLEU and BERTScore results
        st.write(f"**BLEU Score:** {bleu:.4f}")
        st.write(f"**BERTScore Precision:** {bert['precision']:.4f}")
        st.write(f"**BERTScore Recall:** {bert['recall']:.4f}")
        st.write(f"**BERTScore F1:** {bert['f1']:.4f}")
    else:
        st.error("Please provide both reference and generated answers.")
