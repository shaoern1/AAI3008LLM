import streamlit as st

st.set_page_config(page_title="Setup", page_icon="📈")

st.markdown("# ⚙️ Setup your API and Connections")
st.sidebar.header("Upload Documents")
st.sidebar.markdown("""
                    ###Setup your connections here and relevant API keys and Endpoints
                    """)

# After setting the apis and stuff it should save .env file that we can use read from other pages!!