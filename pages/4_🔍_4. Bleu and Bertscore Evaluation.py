# streamlit_app.py
import streamlit as st
import os
from dotenv import load_dotenv
from evaluation import compute_bleu, compute_sacrebleu, compute_bert_score

# Load environment variables at the start
load_dotenv()

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
        # Compute BLEU
        bleu_nltk = compute_bleu(reference, candidate)
        bleu_sacre = compute_sacrebleu(reference, candidate)

        # Display BLEU results
        st.write(f"**BLEU Score (NLTK):** {bleu_nltk:.4f}")
        st.write(f"**BLEU Score (SacreBLEU):** {bleu_sacre:.4f}")

        # Load and Compute Bert Score if User want to,
        bert = compute_bert_score(reference, candidate)

        if bert is not None:
        #Compute and display Bert Score
            st.write(f"**BERTScore Precision:** {bert['precision']:.4f}")
            st.write(f"**BERTScore Recall:** {bert['recall']:.4f}")
            st.write(f"**BERTScore F1:** {bert['f1']:.4f}")

    else:
        st.error("Please provide both reference and generated answers.")