import streamlit as st
import time
import os
from dotenv import load_dotenv
import VectorStore as VSPipe
import AgenticLLM # DEPRECATED
import agent as RAGPipe
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain.agents import Tool, initialize_agent, AgentType
from functools import partial

# Load environment variables
load_dotenv()

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

# ::: Can use session_state from streamlit as Global Vars
client = VSPipe.setup_Qdrant_client()
collection_list = VSPipe.get_collection_names(client)

# Model name
model_path = st.text_input("Enter your Model Name:", "phi4-mini")
collection_select = st.selectbox("Select your collection", collection_list)
st.write(f'Current Model Selected: {model_path}')
st.write(f'Current Vector DB Selected {collection_select}')

# Initialize session state variables if they don't exist
if 'agent_loaded' not in st.session_state:
    st.session_state.agent_loaded = False
if 'rag_agent' not in st.session_state:  # Check for 'rag_agent' instead of 'agent'
    st.session_state.rag_agent = None

# Button to load the agent
if not st.session_state.agent_loaded:
    if st.button("Load Agent"):
        with st.spinner("Initializing agent..."):
            try:
                # Initialize Agent
                st.session_state.rag_agent = RAGPipe.RAGAgent(
                    client=client,  
                    collection_name=collection_select,
                    llm_model=model_path
                )
                st.session_state.agent_loaded = True
                st.success("Agent successfully loaded!")
            except Exception as e:
                st.error(f"Error loading agent: {str(e)}")
else:
    # Option to reset agent
    if st.button("Reset Agent"):
        st.session_state.agent_loaded = False
        st.session_state.rag_agent = None
        st.rerun()

# Only show search options if agent is loaded
if st.session_state.agent_loaded:
    user_query = st.text_input("Enter your query:", "")
    enable_online_search = st.checkbox("Enable Online Search", value=False)
    
    # Button to trigger search
    if st.button("Search"):
        if user_query:
            with st.spinner("Retrieving information..."):
                try:
                    response = st.session_state.rag_agent.invoke(user_query, enable_online_search)
                    st.write("## Response:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
        else:
            st.warning("Please enter a query first")
