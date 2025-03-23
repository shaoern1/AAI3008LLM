import streamlit as st
import os
from dotenv import load_dotenv
import VectorStore as VSPipe
import agent as RAGPipe
import speech_recognition as sr

# Load environment variables
load_dotenv()

st.set_page_config(page_title="RAG Search", page_icon="🔍")

st.markdown("# 🔍 RAG Search")
st.sidebar.header("Settings")

if not os.path.exists(".env"):
    st.error("Please setup your API and Connections first")
    st.stop()

# Initialize Vector DB client
client = VSPipe.setup_Qdrant_client()
collection_list = VSPipe.get_collection_names(client)

# Sidebar settings
model_path = st.sidebar.text_input("Enter Model Name:")
collection_select = st.sidebar.selectbox("Select a Collection", collection_list)
enable_online_search = st.sidebar.checkbox("Enable Online Search", value=False)

# Initialize session state variables
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # Store chat history
if "loaded_model" not in st.session_state:
    st.session_state.loaded_model = None  # Track last loaded model
if "error" not in st.session_state:
    st.session_state.error = None  # Track loading errors

# Button to load agent
if st.sidebar.button("Load Agent"):
    if not model_path.strip():  # Check if the model name is empty
        st.session_state.error = "⚠️ Please enter a model name before loading."
    else:
        with st.spinner("Initializing agent..."):
            try:
                st.session_state.agent = RAGPipe.RAGAgent(
                    client=client, 
                    collection_name=collection_select, 
                    llm_model=model_path
                )
                st.session_state.loaded_model = model_path  # Update model tracker
                st.session_state.messages = []  # Clear chat history on model change
                st.session_state.error = None  # Clear any previous errors
                st.success("Agent loaded successfully!")
            except Exception as e:
                st.session_state.agent = None
                st.session_state.error = f"❌ Error loading agent: {str(e)}"

# Display error message if any
if st.session_state.error:
    st.error(st.session_state.error)

# Only show chat if agent is loaded
if st.session_state.agent:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input field for user messages
    user_input = st.chat_input("Ask me anything...")

    # Add speech-to-text functionality for voice input
    def recognize_speech():
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        with sr.Microphone() as source:
            st.info("Listening...")
            audio = recognizer.listen(source)
            try:
                # Recognize speech using Google Speech Recognition
                text = recognizer.recognize_google(audio)
                st.success(f"Recognized Speech: {text}")
                return text
            except sr.UnknownValueError:
                st.error("Could not understand the audio.")
            except sr.RequestError:
                st.error("Could not request results from Google Speech Recognition service.")
            return ""

    # Button to trigger speech recognition
    if st.button("🎤 Use Voice Input"):
        voice_input = recognize_speech()
        if voice_input:
            user_input = voice_input

    if user_input:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Generate AI response
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent.invoke(user_input, enable_online_search)
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.write(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.warning("⚠️ Please load a model to start chatting.")