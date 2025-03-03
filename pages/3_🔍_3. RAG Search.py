import streamlit as st
import time
import os
from dotenv import load_dotenv
import AgenticLLM as RAGPipe  # Import the RAG pipeline

st.set_page_config(page_title="RAG Search", page_icon="🔍")

st.markdown("# Retrieve Answers with RAG")
st.sidebar.header("RAG Search Interface")
st.write("""
This page allows you to retrieve information using a Retrieval-Augmented Generation (RAG) model.
""")

# Check if environment file exists
if not os.path.exists('.env'):
    st.error("Please setup your API and Connections first")
    st.stop()

# Load environment variables
load_dotenv()

# Input query
query = st.text_input("Enter your query:", "")

# Option to enable online search
enable_online_search = st.checkbox("Enable Online Search", value=False)

# Button to trigger search
if st.button("Search"):
    with st.spinner("Retrieving information..."):
        response = RAGPipe.agent_search(query)  # Call the agent search function
        st.write("## Response:")
        st.write(response)