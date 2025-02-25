import streamlit as st

st.set_page_config(page_title="Setup", page_icon="📈")

st.markdown("# ⚙️ Setup your API and Connections")
st.sidebar.header("Upload Documents")
st.sidebar.markdown("""
                    ### Setup your connections here and relevant API keys and Endpoints
                    """)

# Qdrant API, Qdrant Vector Store API, google search API, google CSE API
qdrant_api = st.text_input("Qdrant API", "http://localhost:6333")
qdranturl_api = st.text_input("Qdrant URL", "http://localhost:6333")
google_search_api = st.text_input("Google Search API", "")
google_cse_api = st.text_input("Google CSE API", "")

# Save the API keys and Endpoints to .env file after pressing the button
if st.button("Save API Keys"):
    with open('.env', 'w') as f:
        f.write(f"QDRANT_API={qdrant_api}\n")
        f.write(f"QDRANT_URL={qdranturl_api}\n")
        f.write(f"GOOGLE_API_KEY={google_search_api}\n")
        f.write(f"GOOGLE_CSE_API={google_cse_api}\n")
    st.success("API Keys and Endpoints saved to .env file")




# After setting the apis and stuff it should save .env file that we can use read from other pages!!