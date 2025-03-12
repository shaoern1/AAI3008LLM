import streamlit as st
import os
from dotenv import load_dotenv
import VectorStore as VSPipe
import agent as RAGPipe

load_dotenv()

st.set_page_config(page_title="RAG Search", page_icon="🔍")

st.markdown("# Retrieve Answers with RAG")
st.sidebar.header("RAG Search Interface")

if not os.path.exists('.env'):
    st.error("Please setup your API and Connections first")
    st.stop()

client = VSPipe.setup_Qdrant_client()
collection_list = VSPipe.get_collection_names(client)

model_path = st.text_input("Enter your Model Name:", "mistral")
collection_select = st.selectbox("Select your collection", collection_list)

st.write(f'Current Model: {model_path}')
st.write(f'Current Vector DB: {collection_select}')

if 'agent_loaded' not in st.session_state:
    st.session_state.agent_loaded = False
if 'rag_agent' not in st.session_state:
    st.session_state.rag_agent = None

if not st.session_state.agent_loaded:
    if st.button("Load Agent"):
        with st.spinner("Initializing agent..."):
            try:
                st.session_state.rag_agent = RAGPipe.RAGAgent(client=client, collection_name=collection_select, llm_model=model_path)
                st.session_state.agent_loaded = True
                st.success("Agent successfully loaded!")
            except Exception as e:
                st.error(f"Error loading agent: {str(e)}")
else:
    if st.button("Reset Agent"):
        st.session_state.agent_loaded = False
        st.session_state.rag_agent = None
        st.rerun()

if st.session_state.agent_loaded:
    user_query = st.text_input("Enter your query:")
    enable_online_search = st.checkbox("Enable Online Search", value=False)

    if st.button("Search"):
        if user_query:
            with st.spinner("Retrieving information..."):
                try:
                    response, bert_score = st.session_state.rag_agent.invoke(user_query, enable_online_search)
                    st.write("## Response:")
                    st.write(response)

                    # Display BERTScore
                    st.write("### Evaluation (BERTScore):")
                    st.write(f"**Similarity Score:** {bert_score}")

                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
        else:
            st.warning("Please enter a query first")
