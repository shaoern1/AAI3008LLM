import streamlit as st
import json
import pandas as pd
import numpy as np
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# Import your RAG agent
import sys
sys.path.append('.')  # Add current directory to path
import VectorStore as VSPipe
from agent import RAGAgent
import nltk
# Add BERTScore
from bert_score import score as bert_score


# Initialize page configuration
st.set_page_config(
    page_title="RAG Evaluation",
    page_icon="📊",
    layout="wide"
)

# Basic page styling
st.markdown("""
<style>
    .metric-row {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .question {
        font-weight: bold;
        font-size: 16px;
    }
    .comparison {
        display: flex;
        margin-top: 5px;
    }
    .reference, .prediction {
        padding: 10px;
        background-color: #f9f9f9;
        border-radius: 5px;
        margin-right: 10px;
        flex: 1;
    }
    .metrics {
        display: flex;
        margin-top: 5px;
    }
    .metric {
        margin-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

def load_qa_pairs(file_path="QA.json"):
    """Load QA pairs from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading QA pairs: {str(e)}")
        return []

def calculate_bert_score(predictions, references):
    """Calculate BERTScore for the predictions and references."""
    try:
        # Using the default model ('roberta-large')
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
        # Convert torch tensors to numpy arrays
        P = P.numpy()
        R = R.numpy()
        F1 = F1.numpy()
        return P, R, F1
    except Exception as e:
        st.warning(f"Error calculating BERTScore: {str(e)}")
        # Return empty arrays of the correct length
        return (np.zeros(len(predictions)), 
                np.zeros(len(predictions)), 
                np.zeros(len(predictions)))

@st.cache_data
def evaluate_rag_model(qa_pairs, model, collection, web_search=False):
    """Evaluate RAG model responses against reference answers."""
    # Initialize metrics
    rouge = Rouge()
    smooth = SmoothingFunction().method1
    
    # Initialize results dict
    results = {
        'questions': [],
        'references': [],
        'predictions': [],
        'bleu_1': [],
        'bleu_4': [],
        'rouge_1_f': [],
        'rouge_l_f': [],
        'response_time': [],
        'bert_score_p': [],
        'bert_score_r': [],
        'bert_score_f1': []
    }
    
    # Initialize RAG agent
    client = VSPipe.setup_Qdrant_client()
    agent = RAGAgent(client=client, collection_name=collection, llm_model=model)
    
    # Process each question
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_references = []
    all_predictions = []
    
    for i, pair in enumerate(qa_pairs):
        question = pair['question']
        reference = pair['answer']
        
        # Update status
        status_text.text(f"Processing question {i+1}/{len(qa_pairs)}: {question[:50]}...")
        
        # Get model prediction with timing
        start_time = time.time()
        try:
            prediction = agent.invoke(question, enable_search=web_search)
        except Exception as e:
            st.error(f"Error getting prediction for question: {question}")
            st.error(f"Error message: {str(e)}")
            prediction = f"ERROR: {str(e)}"
        end_time = time.time()
        response_time = end_time - start_time
        
        # Calculate BLEU scores
        try:
            reference_tokens = nltk.word_tokenize(reference.lower())
            prediction_tokens = nltk.word_tokenize(prediction.lower())
            
            bleu_1 = sentence_bleu([reference_tokens], prediction_tokens, 
                                weights=(1, 0, 0, 0), smoothing_function=smooth)
            bleu_4 = sentence_bleu([reference_tokens], prediction_tokens, 
                                weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        except Exception as e:
            bleu_1 = bleu_4 = 0
        
        # Calculate ROUGE scores
        try:
            rouge_scores = rouge.get_scores(prediction, reference)[0]
            rouge_1_f = rouge_scores['rouge-1']['f']
            rouge_l_f = rouge_scores['rouge-l']['f']
        except Exception as e:
            rouge_1_f = rouge_l_f = 0
        
        # Store results
        results['questions'].append(question)
        results['references'].append(reference)
        results['predictions'].append(prediction)
        results['bleu_1'].append(bleu_1)
        results['bleu_4'].append(bleu_4)
        results['rouge_1_f'].append(rouge_1_f)
        results['rouge_l_f'].append(rouge_l_f)
        results['response_time'].append(response_time)
        
        # Collect references and predictions for batch BERTScore calculation
        all_references.append(reference)
        all_predictions.append(prediction)
        
        # Update progress
        progress_bar.progress((i + 1) / len(qa_pairs))
    
    # Calculate BERTScore in batch for better performance
    status_text.text("Calculating BERTScore...")
    P, R, F1 = calculate_bert_score(all_predictions, all_references)
    results['bert_score_p'] = P.tolist()
    results['bert_score_r'] = R.tolist()
    results['bert_score_f1'] = F1.tolist()
    
    status_text.text("Evaluation complete!")
    
    # Calculate average scores
    results['avg_bleu_1'] = np.mean(results['bleu_1'])
    results['avg_bleu_4'] = np.mean(results['bleu_4'])
    results['avg_rouge_1_f'] = np.mean(results['rouge_1_f'])
    results['avg_rouge_l_f'] = np.mean(results['rouge_l_f'])
    results['avg_response_time'] = np.mean(results['response_time'])
    results['avg_bert_score_f1'] = np.mean(results['bert_score_f1'])
    
    return results

def main():
    st.title("RAG Evaluation")
    
    # Check if bert-score is installed
    try:
        import bert_score
        bert_score_available = True
    except ImportError:
        st.warning("BERTScore is not installed. Run: `pip install bert-score` to enable this metric.")
        st.markdown("""
        ```
        pip install bert-score
        ```
        """)
        bert_score_available = False
    
    # Load QA pairs
    qa_pairs = load_qa_pairs()
    client = VSPipe.setup_Qdrant_client()
    collection_list = VSPipe.get_collection_names(client)

    # Display basic configuration options
    col1, col2, col3 = st.columns(3)
    with col1:
        model_options = ["phi3:mini", "gemma3:4b", "phi4-mini"]
        model = st.selectbox("Select Model", model_options)
    
    with col2:
        collection_select = st.selectbox("Select your collection", collection_list)

    
    # Start evaluation button
    if st.button("Start Evaluation"):
        with st.spinner("Evaluating RAG model..."):
            results = evaluate_rag_model(qa_pairs, model, collection_select, False)
        
        # Display summary metrics
        st.subheader("Summary Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Average BLEU-1", f"{results['avg_bleu_1']:.3f}")
        col2.metric("Average BLEU-4", f"{results['avg_bleu_4']:.3f}")
        col3.metric("Average ROUGE-L", f"{results['avg_rouge_l_f']:.3f}")
        col4.metric("Average BERTScore F1", f"{results['avg_bert_score_f1']:.3f}" if bert_score_available else "N/A")
        col5.metric("Avg Response Time", f"{results['avg_response_time']:.2f}s")
        
        if bert_score_available:
            metrics = ['BLEU-1', 'BLEU-4', 'ROUGE-L', 'BERTScore-F1']
            values = [
                results['avg_bleu_1'],
                results['avg_bleu_4'],
                results['avg_rouge_l_f'],
                results['avg_bert_score_f1']
            ]

            # Create a DataFrame with your metrics as the index
            # Use pd.Categorical to enforce the order
            df = pd.DataFrame({'Score': values}, index=pd.Categorical(metrics, categories=metrics, ordered=True))

            # Create bar chart with proper y-axis limits
            st.bar_chart(df, y_label='Avg Score', height=400)

        
        # Display per-question results
        st.subheader("Per-Question Results")
        
        for i in range(len(results['questions'])):
            with st.container():
                st.markdown(f"### Question {i+1}")
                st.markdown(f"**{results['questions'][i]}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Reference Answer:**")
                    st.markdown(f"{results['references'][i]}")
                
                with col2:
                    st.markdown("**Generated Answer:**")
                    st.markdown(f"{results['predictions'][i]}")
                
                # Display metrics
                st.markdown("**Metrics:**")
                if bert_score_available:
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("BLEU-1", f"{results['bleu_1'][i]:.3f}")
                    col2.metric("BLEU-4", f"{results['bleu_4'][i]:.3f}")
                    col3.metric("ROUGE-L", f"{results['rouge_l_f'][i]:.3f}")
                    col4.metric("BERTScore-F1", f"{results['bert_score_f1'][i]:.3f}")
                    col5.metric("Response Time", f"{results['response_time'][i]:.2f}s")
                else:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("BLEU-1", f"{results['bleu_1'][i]:.3f}")
                    col2.metric("BLEU-4", f"{results['bleu_4'][i]:.3f}")
                    col3.metric("ROUGE-L", f"{results['rouge_l_f'][i]:.3f}")
                    col4.metric("Response Time", f"{results['response_time'][i]:.2f}s")
                
                st.markdown("---")
        
        # Create a simple exportable report
        report_data = {
            'Question': results['questions'],
            'Reference': results['references'],
            'Generated': results['predictions'],
            'BLEU-1': [f"{score:.3f}" for score in results['bleu_1']],
            'BLEU-4': [f"{score:.3f}" for score in results['bleu_4']],
            'ROUGE-1': [f"{score:.3f}" for score in results['rouge_1_f']],
            'ROUGE-L': [f"{score:.3f}" for score in results['rouge_l_f']],
        }
        
        if bert_score_available:
            report_data['BERTScore-P'] = [f"{score:.3f}" for score in results['bert_score_p']]
            report_data['BERTScore-R'] = [f"{score:.3f}" for score in results['bert_score_r']]
            report_data['BERTScore-F1'] = [f"{score:.3f}" for score in results['bert_score_f1']]
        
        df = pd.DataFrame(report_data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="rag_evaluation_results.csv",
            mime="text/csv"
        )
    else:
        # Show instructions when not running evaluation
        st.info("Click 'Start Evaluation' to begin the evaluation process.")

if __name__ == "__main__":
    main()