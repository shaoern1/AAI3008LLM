import streamlit as st

import nltk


nltk.download('punkt_tab')

st.set_page_config(
page_title="Home",
page_icon="👋",
)
  
st.sidebar.success("Select a Page Above")

st.markdown(
    """
    ### olqRAG is a simple open-source RAG LLM system using Cloud Vector Storage and Localhost LLMs. 
    For any RAG LLMs use-case running on your very own machines. Built using Ollama, Langchain, Qdrant,.
    
    **👈** Select the headers on the left to start.
    
    ### Instructions
    1. Setup Environments and Connections
    2. Upload Document into Vector Database
    3. Select Ollama Model and Run your queries
    4. Perform Evaluation on various models
    
    ### Features and Support
    Supports Google Search
    """
)