import streamlit as st
import time
import numpy as np
import VectorStore as VSPipe
import os
import dotenv


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
# collection name input     
collection_name = st.text_input("Collection Name", "Ollama-RAG")

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

    # Load and process the document
    print(f"Loading document: {file.name}")
    documents = VSPipe.load_document(file.name)

    print(f"Creating chunks from document")
    chunks = VSPipe.create_chunks(documents)
    print(f"Created {len(chunks)} chunks")

    print("Generating embeddings (this may take some time)...")

    # Generate embeddings
    embeddingsList = VSPipe.generateEmbeddings(chunks)
    # Prepare vector points
    embeddingsList = VSPipe.prepare_vector_points(embeddingsList)

    # Upsert points
    collection_name = VSPipe.create_collection()
    VSPipe.upsert_points(VSPipe.client, collection_name, embeddingsList)

    st.write("Embeddings generated and uploaded successfully")
    st.write("Collection name: ", collection_name)
    st.write("Number of embeddings: ", len(embeddingsList))

    st.write("Done")

