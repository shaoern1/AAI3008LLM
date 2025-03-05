import streamlit as st
import time
import numpy as np
import VectorStore as VSPipe
import os
from dotenv import load_dotenv

# Initialize session state for tracking which input was last modified
if 'last_modified' not in st.session_state:
    st.session_state.last_modified = None
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = ""

# Callback functions to track changes
def on_input_change():
    st.session_state.last_modified = "input"
    st.session_state.collection_name = st.session_state.collection_input

def on_select_change():
    if st.session_state.collection_select != '':
        st.session_state.last_modified = "select"
        st.session_state.collection_name = st.session_state.collection_select

st.set_page_config(page_title="Upload Documents", page_icon="📈")

st.markdown("# Upload Documents to Vector Database")
st.sidebar.header("Upload Documents")
st.write(
    """Add description of the page here."""
)

# check if environment file exists
if not os.path.exists('.env'):
    st.error("Please setup your API and Connections first")
    st.stop()
    
# Setup Qdrant Client
client = VSPipe.setup_Qdrant_client()
default_collection_list = ['']
    
# collection name input     
collection_input = st.text_input("Enter a new collection name:", 
                               value="", 
                               key="collection_input", 
                               on_change=on_input_change)

st.write("**OR**")
collection_list = VSPipe.get_collection_names(client)
default_collection_list.extend(collection_list)

collection_select = st.selectbox("Select an existing collection", 
                               options=default_collection_list, 
                               key="collection_select", 
                               on_change=on_select_change)

st.write(f"**Current collection:** {st.session_state.collection_name}")

file = st.file_uploader("Upload a file", type=["pdf", "txt", "md"])
if file is not None:
    st.write("File uploaded successfully")
    st.write(file.name)
    st.write(file.type)
    st.write(file.size)
    
    # Save the uploaded file temporarily
    with open(file.name, "wb") as f:
        f.write(file.getbuffer())
    
    # Pass the file path to your function
    file_path = file

#button to start loading and processing
if st.button("Load and Process Document"):

    # Load env
    load_dotenv()
    
    # Load and process the document
    print(f"Loading document: {file.name}")
    documents = VSPipe.load_document(file.name)

    print(f"Creating chunks from document")
    chunks = VSPipe.create_chunks(documents)
    print(f"Created {len(chunks)} chunks")

    print("Generating embeddings (this may take some time)...")

    # Generate embeddings
    try:
        embeddingsList = VSPipe.generateEmbeddings(chunks)
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}. Please refer to documentations.")
        st.stop()  # Stop execution if there's an error
        
    
    # Prepare vector points
    embeddingsList = VSPipe.prepare_vector_points(embeddingsList)
    
    # Upsert points
    if st.session_state.last_modified == "input":
        collection_name = st.session_state.collection_name
        collection_name = VSPipe.create_collection(client, collection_name)
    else:
        collection_name = st.session_state.collection_name
    
    VSPipe.upsert_points(client, collection_name, embeddingsList)

    st.write("Embeddings generated and uploaded successfully")
    st.write("Collection name: ", collection_name)
    st.write("Number of embeddings: ", len(embeddingsList))

    st.write("Done")

