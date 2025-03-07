import streamlit as st
import time
import os
from dotenv import load_dotenv
import VectorStore as VSPipe
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

# ::: Can use session_state from streamlit as Global Vars
client = VSPipe.setup_Qdrant_client()
collection_list = VSPipe.get_collection_names(client)

# Model name
model_path = st.text_input("Enter your Model Name:", "phi3:mini")
collection_select = st.selectbox("Select your collection", collection_list)
st.write(f'Current Model Selected: {model_path}')
st.write(f'Current Vector DB Selected {collection_select}')

# Initialize session state variables if they don't exist
if 'agent_loaded' not in st.session_state:
    st.session_state.agent_loaded = False
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'agent_with_search' not in st.session_state:
    st.session_state.agent_with_search = None
if 'last_search_setting' not in st.session_state:
    st.session_state.last_search_setting = None

# Button to load the agent
if not st.session_state.agent_loaded:
    if st.button("Load Agent"):
        with st.spinner("Initializing agent..."):
            # Initialize Agent
            st.session_state.rag_agent = RAGPipe.RAGAgent(
                client=client,  
                collection_name=collection_select,
                llm_model=model_path
            )
            
            # Initialize with default setting (no search)
            st.session_state.agent = st.session_state.rag_agent.init_agent(enable_search=False)
            st.session_state.last_search_setting = False
            st.session_state.agent_loaded = True
            
            try:
                # Run warmup with error handling
                st.session_state.agent.invoke({"input": "Hello"})  # Warmup
            except Exception as e:
                # Silently ignore warmup errors
                st.error(f"Warmup error (this can be ignored): {str(e)}")
                pass
            
            # Show confirmation
            st.success("Agent successfully loaded!")

# Only show search options if agent is loaded
if st.session_state.agent_loaded:
    # Option to enable online search
    # Input query
    user_query = st.text_input("Enter your query:", "")
    enable_online_search = st.checkbox("Enable Online Search", value=False)
    
    # Button to trigger search
    if st.button("Search"):
        if user_query:
            with st.spinner("Retrieving information..."):
                prompt_info = st.session_state.rag_agent.get_formatted_prompt(
                    user_query, 
                    enable_search=enable_online_search
                )
                
                st.write("## Complete Prompt Template:")
                st.text(prompt_info["formatted_prompt"])
                
                st.write("## Individual Components:")
                st.write("### Context:")
                st.text(prompt_info["context"])
                
                st.write("### Online Context:")
                st.text(prompt_info["online_context"])

                try:
                    # Check if we need to create a new agent with different search settings
                    if st.session_state.last_search_setting != enable_online_search:
                        st.session_state.agent = st.session_state.rag_agent.init_agent(enable_search=enable_online_search)
                        st.session_state.last_search_setting = enable_online_search
                    
                    # Use the agent with the correct search settings
                    response = st.session_state.agent.invoke({"input": user_query})
                    st.write("## Response:")
                    if isinstance(response, dict) and "output" in response:
                        st.write(response["output"])  # Just write the output if it's in expected format (dict with "output" key)   
                    else:
                        st.write(response)  # Just write the whole response if it's not in expected format
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.error("Try refreshing the page and loading the agent again.")
        else:
            st.warning("Please enter a query first")