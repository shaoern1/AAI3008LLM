# streamlit_app.py
import sys
import os
import traceback

# Set up more detailed logging first
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting app with enhanced debugging")

# Import the debug module to apply patches before any other imports
try:
    import streamlit_debug
    streamlit_debug.apply_patches()
    logger.info("Applied debug patches successfully")
except ImportError:
    logger.warning("Debug module not found, continuing without patches")

# Now continue with regular imports
try:
    logger.info("Importing streamlit")
    import streamlit as st
    logger.info("Streamlit imported successfully")
except Exception as e:
    logger.critical(f"Failed to import streamlit: {e}")
    logger.critical(traceback.format_exc())
    print(f"CRITICAL ERROR: Cannot import streamlit: {e}")
    sys.exit(1)
    
try:
    logger.info("Importing dotenv")
    from dotenv import load_dotenv
    logger.info("Dotenv imported successfully")
except Exception as e:
    logger.error(f"Failed to import dotenv: {e}")
    logger.error(traceback.format_exc())

# Try to import each evaluation function separately to isolate issues
try:
    logger.info("Importing compute_bleu")
    from evaluation import compute_bleu
    logger.info("compute_bleu imported successfully")
except Exception as e:
    logger.error(f"Failed to import compute_bleu: {e}")
    logger.error(traceback.format_exc())

try:
    logger.info("Importing compute_sacrebleu")
    from evaluation import compute_sacrebleu
    logger.info("compute_sacrebleu imported successfully")
except Exception as e:
    logger.error(f"Failed to import compute_sacrebleu: {e}")
    logger.error(traceback.format_exc())

try:
    logger.info("Importing compute_bert_score")
    from evaluation import compute_bert_score
    logger.info("compute_bert_score imported successfully")
except Exception as e:
    logger.error(f"Failed to import compute_bert_score: {e}")
    logger.error(traceback.format_exc())

# Load environment variables
try:
    logger.info("Loading environment variables")
    load_dotenv()
    logger.info("Environment variables loaded")
except Exception as e:
    logger.error(f"Failed to load environment variables: {e}")
    logger.error(traceback.format_exc())

try:
    logger.info("Setting page config")
    st.set_page_config(
        page_title="Evaluate Model Response",
        page_icon="📝",
    )
    logger.info("Page config set successfully")
except Exception as e:
    logger.error(f"Failed to set page config: {e}")
    logger.error(traceback.format_exc())

try:
    logger.info("Setting up markdown header")
    st.markdown("## Evaluate Model Response")
    logger.info("Markdown header set successfully")
except Exception as e:
    logger.error(f"Failed to set markdown header: {e}")
    logger.error(traceback.format_exc())

# Input fields for reference and generated answers
try:
    logger.info("Creating text areas")
    reference = st.text_area("Reference Answer", "")
    candidate = st.text_area("Generated Answer", "")
    logger.info("Text areas created successfully")
except Exception as e:
    logger.error(f"Failed to create text areas: {e}")
    logger.error(traceback.format_exc())

try:
    logger.info("Setting up evaluate button")
    evaluate_button = st.button("Evaluate")
    logger.info(f"Button created: {evaluate_button}")
except Exception as e:
    logger.error(f"Failed to create button: {e}")
    logger.error(traceback.format_exc())

try:
    logger.info("Setting up button logic")
    if evaluate_button:
        logger.info("Evaluate button clicked")
        if reference and candidate:
            logger.info("Both reference and candidate text provided")
            
            try:
                # Compute BLEU
                logger.info("Computing BLEU scores")
                bleu_nltk = compute_bleu(reference, candidate)
                logger.info(f"NLTK BLEU computed: {bleu_nltk}")
                
                bleu_sacre = compute_sacrebleu(reference, candidate)
                logger.info(f"SacreBLEU computed: {bleu_sacre}")

                # Display BLEU results
                st.write(f"**BLEU Score (NLTK):** {bleu_nltk:.4f}")
                st.write(f"**BLEU Score (SacreBLEU):** {bleu_sacre:.4f}")

                # Try to compute BERTScore
                logger.info("Computing BERTScore")
                bert = compute_bert_score(reference, candidate)
                logger.info(f"BERTScore computed: {bert}")
                
                if bert is not None:
                    # Display BERTScore results
                    st.write(f"**BERTScore Precision:** {bert['precision']:.4f}")
                    st.write(f"**BERTScore Recall:** {bert['recall']:.4f}")
                    st.write(f"**BERTScore F1:** {bert['f1']:.4f}")
                else:
                    logger.warning("BERTScore returned None")
                    st.warning("BERTScore computation returned None")
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                logger.error(traceback.format_exc())
                st.error(f"An error occurred during evaluation: {str(e)}")
        else:
            logger.info("Missing reference or candidate text")
            st.error("Please provide both reference and generated answers.")
    else:
        logger.info("Showing initial info message")
        st.info("Click the 'Evaluate' button to compute scores.")
except Exception as e:
    logger.critical(f"Fatal error in button logic: {e}")
    logger.critical(traceback.format_exc())
    st.error(f"A critical error occurred: {str(e)}")

logger.info("App script completed")