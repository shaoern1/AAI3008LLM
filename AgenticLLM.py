import os
import uuid
import warnings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from langchain_qdrant import FastEmbedSparse, Qdrant
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
import nest_asyncio
from llama_parse import LlamaParse
import ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_google_community import GoogleSearchAPIWrapper

# Enable asyncio nest
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Qdrant setup
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
collection_name = "Ollama-RAG"

qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

vectorstore = Qdrant(
    client=qdrant_client,
    collection_name=collection_name,
    embeddings=OllamaEmbeddings,  # Handling embeddings separately
)

def hybrid_search(query: str, collection: str, limit: int = 5):
    dense_vector = ollama.embeddings(model='rjmalagon/gte-qwen2-1.5b-instruct-embed-f16', prompt=query)
    
    results = qdrant_client.search(
        collection_name=collection,
        query_vector=models.NamedVector(
            name="gte-qwen1.5",
            vector=dense_vector['embedding']
        ),
        limit=limit,
        search_params=models.SearchParams(
            hnsw_ef=128
        )
    )
    return results

def retrieve_documents(query):
    results = hybrid_search(query, collection=collection_name, limit=3)
    context = "Database Context: "
    for result in results:
        context += str(result.payload['text'])
    return context

def get_rag_chain(enable_search=False):
    """Creates and returns the RAG chain with hybrid search and optional Google Search."""
    llm = Ollama(model="phi3:mini")
    
    template = """Answer the question based on the following context:
    {context}
    {online_context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    def online_search(query):
        search = GoogleSearchAPIWrapper()
        search_results = search.run(query)
        return f"Online search results: {search_results}" if search_results else "No relevant information found online."
    
    def filler(_):
        return "Online Search Switched Off"
    
    rag_chain = (
        {
            "context": retrieve_documents,
            "question": RunnablePassthrough(),
            "online_context": online_search if enable_search else filler
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def agent_search(query):
    """Search for relevant information in the document and online."""
    rag_chain = get_rag_chain(enable_search=False)
    return rag_chain.invoke(query)

# Define tools for the agent
tools = [
    Tool(
        name="document_search",
        func=agent_search,
        description="Useful for answering questions based on document and online search."
    )
]

# Initialize agent
llm = Ollama(model="phi3:mini")
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Suppress warnings
warnings.filterwarnings('ignore')


