import os
import uuid
import warnings
import ollama
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from langchain_qdrant import FastEmbedSparse, Qdrant
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
import nest_asyncio
from llama_parse import LlamaParse
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_google_community import GoogleSearchAPIWrapper
from functools import partial
from langchain_community.llms import Ollama  # or another local LLM wrapper
from langchain.callbacks.tracers import LangChainTracer
from langsmith import Client


tracer = LangChainTracer(project_name="AAILLMPROJ")


# Enable asyncio nest
def enable_asyncio():
    nest_asyncio.apply()

# Class-based implementation of RAG functionality 
# CLASS IMPLEMENTATION WAS TO AVOID PARAMETER DRILLING
class RAGAgent:
    def __init__(self, client, collection_name, llm_model="phi3:mini", 
                 dense_model='rjmalagon/gte-qwen2-1.5b-instruct-embed-f16'):
        """
        Initialize the RAG agent with the required parameters
        
        Args:
            client: Qdrant client instance
            collection_name: Name of the collection to search
            llm_model: Name of the LLM model to use
            dense_model: Name of the embedding model to use
        """
        self.client = client
        self.collection_name = collection_name
        self.llm_model = llm_model
        self.dense_model = dense_model
        self.llm = ChatOllama(model=llm_model, callbacks=[tracer])  
    
    def hybrid_search(self, query: str, limit: int = 5):
        """
        Perform hybrid search on the vector database
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of search results
        """
        dense_vector = ollama.embeddings(model=self.dense_model, prompt=query)
        
        results = self.client.search(
            collection_name=self.collection_name,
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

    def retrieve_documents(self, query):
        """
        Retrieve documents based on the query
        
        Args:
            query: The search query
            
        Returns:
            String context from retrieved documents
        """
        results = self.hybrid_search(query, limit=5)
        context = "Database Context: "
        for result in results:
            context += str(result.payload['text'])
        return context

    def online_search(self, query):
        """Perform online search using Google"""
        search = GoogleSearchAPIWrapper()
        search_results = search.run(query)
        return f"Online search results: {search_results}" if search_results else "No relevant information found online."
    
    def filler(self, _):
        """Placeholder when online search is disabled"""
        return "Online Search Switched Off"

    def get_rag_chain(self, enable_search=False):
        """
        Create and return the RAG chain
        
        Args:
            enable_search: Whether to enable online search
            
        Returns:
            Configured LangChain RAG chain
        """
        template = """Answer the question based on the following context:
        {context}
        
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        rag_chain = (
            {
                "context": self.retrieve_documents,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain
    
    def get_formatted_prompt(self, query, enable_search=False):
        """
        Get the fully formatted prompt without executing it
        
        Args:
            query: The search query
            enable_search: Whether to include online search results
            
        Returns:
            dict: Dictionary containing context, online_context, question and formatted prompt
        """
        # Manually gather all inputs
        context = self.retrieve_documents(query)
        online_context = self.online_search(query) if enable_search else self.filler(query)
        
        # Format the template
        template = """Answer the question based on the following context:
        {context}
        {online_context}
        
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        formatted = prompt.format(
            context=context,
            online_context=online_context,
            question=query
        )
        
        return {
            "context": context,
            "online_context": online_context, 
            "question": query,
            "formatted_prompt": formatted
        }

    def search(self, query, enable_search=False):
        """
        Search for information using the RAG chain
        
        Args:
            query: The search query
            enable_search: Whether to enable online search
            
        Returns:
            Response from the LLM
        """
        rag_chain = self.get_rag_chain(enable_search=enable_search)
        return rag_chain.invoke(query, callbacks=[tracer])

    def init_agent(self, enable_search):
        """
        Initialize a LangChain agent with document search capabilities
        
        Returns:
            An initialized LangChain agent
        """
        # Create wrapper functions that don't require additional parameters
        def document_search(query):
            return self.search(query, enable_search=enable_search)
        
        def online_search_tool(query):
            return self.online_search(query)
        
        # Define tools with simple functions
        tools = [
            Tool(
                name="document_search",
                func=document_search,
                description="Useful for answering questions based on documents in your collection."
            )
        ]
        
        # Only add the online search tool if enabled
        if enable_search:
            tools.append(
                Tool(
                    name="online_search",
                    func=online_search_tool,
                    description="Useful for finding current information online."
                )
            )
        
        # Use a more structured prompt with clearer examples
        from langchain.prompts import PromptTemplate
        from langchain.agents import AgentExecutor, create_react_agent
        
        prompt = PromptTemplate.from_template(
            """Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Here is an example of the correct format:
        Question: What is a loss function?
        Thought: I need to search for information about loss functions.
        Action: document_search
        Action Input: definition of loss function and its importance
        Observation: Loss functions measure how far off predictions are from actual values. They're used in training machine learning models.
        Thought: I now know the final answer.
        Final Answer: A loss function measures the difference between predicted values and actual values, guiding the optimization process in machine learning.

        Begin!

        Question: {input}
        {agent_scratchpad}"""
            )
        
        # Create a custom agent with the improved prompt
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        # Create an executor with a higher max iterations to give it more chances
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            callbacks=[tracer],
            max_iterations=10  # Increased from 3 to 5
        )

# Legacy functions that maintain the same interface for backward compatibility, 
def hybrid_search(query: str, client: QdrantClient, collection: str, 
                 limit: int = 5, dense_model='rjmalagon/gte-qwen2-1.5b-instruct-embed-f16'):
    """Legacy function that creates a RAGAgent and calls its hybrid_search method"""
    agent = RAGAgent(client, collection, dense_model=dense_model)
    return agent.hybrid_search(query, limit)

def retrieve_documents(query, client, collection_name):
    """Legacy function that creates a RAGAgent and calls its retrieve_documents method"""
    agent = RAGAgent(client, collection_name)
    return agent.retrieve_documents(query)

def get_rag_chain(client, collection_name, enable_search=False, llm_model="phi3:mini"):
    """Legacy function that creates a RAGAgent and calls its get_rag_chain method"""
    agent = RAGAgent(client, collection_name, llm_model)
    return agent.get_rag_chain(enable_search)

def agent_search(query, client, collection_name, llm_model='phi3:mini'):
    """Legacy function that creates a RAGAgent and calls its search method"""
    agent = RAGAgent(client, collection_name, llm_model)
    return agent.search(query)

def init_agent(client, collection_name, llm_model='phi3:mini'):
    """Legacy function that creates a RAGAgent and calls its init_agent method"""
    agent = RAGAgent(client, collection_name, llm_model)
    return agent.init_agent()

# Warmup agent in PAGE 3.




