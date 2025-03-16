import streamlit as st
import os
from dotenv import load_dotenv

import VectorStore as VSPipe
import agent as RAGPipe
from llm_judge_functions import custom_llm_factory, evaluate_with_llm_judge  

# Load environment variables
load_dotenv()

st.set_page_config(page_title="LLM Judge", page_icon="⚖️")

st.markdown("# Evaluate using LLM Judge")
st.sidebar.header("Settings")
st.write("""
Retrieve answers using a Retrieval-Augmented Generation (RAG) model 
and evaluate them using an LLM Judge.
""")

# Check environment setup
if not os.path.exists(".env"):
    st.error("⚠️ Please setup your API and connections first.")
    st.stop()

# Initialize Qdrant Client
client = VSPipe.setup_Qdrant_client()
collection_list = VSPipe.get_collection_names(client)

# Sidebar inputs
model_path = st.sidebar.text_input("Model Name for Answer Generation:")
judge_model = st.sidebar.text_input("LLM Model for Judging:")
collection_select = st.sidebar.selectbox("Select a Collection", collection_list)
enable_online_search = st.sidebar.checkbox("Enable Online Search", value=False)

# Display current selections
st.sidebar.write(f'**Selected Model:** {model_path}')
st.sidebar.write(f'**LLM Judge Model:** {judge_model}')
st.sidebar.write(f'**Selected Collection:** {collection_select}')

# Initialize session state
if "agent_loaded" not in st.session_state:
    st.session_state.agent_loaded = False
if "rag_agent" not in st.session_state:
    st.session_state.rag_agent = None

# Load Agent Button
if not st.session_state.agent_loaded:
    if st.sidebar.button("Load Agent"):
        with st.spinner("Initializing agent..."):
            try:
                st.session_state.rag_agent = RAGPipe.RAGAgent(
                    client=client, 
                    collection_name=collection_select, 
                    llm_model=model_path
                )
                st.session_state.agent_loaded = True
                st.success("✅ Agent successfully loaded!")
            except Exception as e:
                st.error(f"❌ Error loading agent: {str(e)}")
else:
    if st.sidebar.button("Reset Agent"):
        st.session_state.agent_loaded = False
        st.session_state.rag_agent = None
        st.rerun()

# Show only if agent is loaded
if st.session_state.agent_loaded:
    st.markdown("## 🧐 Enter Query for Evaluation")
    user_query = st.text_area("Enter your query:", "")

    # Validate LLM Judge model
    try:
        custom_llm_factory(judge_model)  # Ensure model is available
        st.success(f"✅ LLM Judge `{judge_model}` is ready.")
    except ValueError:
        st.error(f"❌ Model `{judge_model}` not found. Install using `ollama pull {judge_model}`.")
        st.stop()

    # Search and Evaluate Button
    if st.button("⚖️ Retrieve and Evaluate"):
        if user_query.strip():
            with st.spinner("Retrieving answer..."):
                try:
                    response = st.session_state.rag_agent.invoke(user_query, enable_online_search)
                    st.markdown("### 📌 Retrieved Answer:")
                    st.success(response)
                except Exception as e:
                    st.error(f"❌ Error retrieving answer: {str(e)}")
                    response = None

            if response and "I couldn't find any relevant information" not in response:
                with st.spinner("Evaluating with LLM Judge..."):
                    try:
                        evaluation_results = evaluate_with_llm_judge(user_query, response, judge_model)
                        st.markdown("### 🏆 LLM Judge Evaluation:")
                        st.info(evaluation_results)
                    except Exception as e:
                        st.error(f"❌ Error during evaluation: {str(e)}")
        else:
            st.warning("⚠️ Please enter a query before evaluating.")
