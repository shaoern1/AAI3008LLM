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
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from typing import TypedDict, List, Dict, Any, Annotated
import operator
from langchain_core.messages import BaseMessage
import VectorStore as VSPipe
from dotenv import load_dotenv

# Update the AgentState TypedDict to include a new field
class AgentState(TypedDict):
    messages: List[BaseMessage]
    info_sufficient: bool
    enable_search: bool
    found_db_info: bool  # New field to track if database had useful info

# Create the tools
web_search = DuckDuckGoSearchRun()


class RAGAgent:
    def __init__(self, client, collection_name, llm_model="phi4-mini", 
                 dense_model='rjmalagon/gte-qwen2-1.5b-instruct-embed-f16'):
        self.client = client
        self.collection_name = collection_name
        self.llm_model = llm_model
        self.dense_model = dense_model
        self.llm = ChatOllama(model=llm_model)
        # Initialize the graph when instance is created
        self.graph = self.create_graph()
        
    def hybrid_search(self, query: str, limit: int = 5):
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
        results = self.hybrid_search(query, limit=5)
        context = "Database Context: "
        for result in results:
            context += str(result.payload['text'])
        return context
    
    def vector_search_node(self, state: AgentState):
        """Search vector database and determine if results are useful"""
        messages = state["messages"]
        # Get the last user query
        user_query = next((msg.content for msg in reversed(messages) 
                          if isinstance(msg, HumanMessage)), "")
        
        # Search documents
        results = self.hybrid_search(query=user_query, limit=5)
        
        # Check if we found meaningful results (non-empty or above certain length)
        found_info = False
        context = ""
        for result in results:
            text = str(result.payload['text'])
            context += text
            # Simple heuristic - if we have some substantial content
            if len(text.strip()) > 50:  
                found_info = True
        
        # Add results to messages with appropriate commentary
        if found_info:
            search_message = AIMessage(content=f"Vector Search Context: {context}")
        else:
            search_message = AIMessage(content="I searched the document database but couldn't find any relevant information for your query.")
        
        return {
            "messages": messages + [search_message],
            "found_db_info": found_info
        }
    
    def evaluate_info_node(self, state: AgentState):
        """Evaluate if information is sufficient"""
        messages = state["messages"]
        
        eval_prompt = HumanMessage(content="""
        Based on the information from Vector Search Context, do you have sufficient information to answer the user's question? 
        Respond with SUFFICIENT if you have enough information, or INSUFFICIENT if you need more information, NOTHING ELSE.
        """)
        
        eval_response = self.llm.invoke(messages + [eval_prompt])
        if eval_response.content == "SUFFICIENT":
            is_sufficient = True
        else:
            is_sufficient = False
        
        return {"info_sufficient": is_sufficient, "messages": messages}
    
    def web_search_node(self, state: AgentState):
        """Search the web for additional information"""
        messages = state["messages"]
        
        # Get the last user query
        user_query = next((msg.content for msg in reversed(messages) 
                          if isinstance(msg, HumanMessage)), "")
        
        # Run web search
        web_results = web_search.run(user_query)
        
        # Add results to messages
        web_message = AIMessage(content=f"Online Search Context:\n{web_results}")
        return {"messages": messages + [web_message]}
    
    def final_response_node(self, state: AgentState):
        """Generate final response based only on retrieved information"""
        messages = state["messages"]
        
        final_prompt = HumanMessage(content="""
        Now provide a comprehensive answer to the original question based ONLY on the information provided 
        in the search results above. Do not use any prior knowledge.
        
        YOU MUST INCLUDE ALL THE INFORMATION EXACTLY IN THE PREVIOUS MESSAGES FROM DATABASE SEARCH AND WEB SEARCH INTO TWO SECTIONS IN YOUR RESPONSE,
        VECTOR SEARCH CONTEXT: , ONLINE SEARCH CONTEXT: ,
        IF THERE WAS NO ONLINE SEARCH CONTEXT DO NOT INCLUDE ONLINE SEARCH CONTEXT:
        
        For each piece of information in your answer, indicate which source it came from 
        (database or web search). If the search results don't contain enough information to 
        answer some aspect of the question, explicitly state that you don't have that information.
        """)
        
        final_response = self.llm.invoke(messages + [final_prompt])
        return {"messages": messages + [final_response]}
    
    def insufficient_info_node(self, state: AgentState):
        """Generate a response when information is insufficient and web search is disabled"""
        messages = state["messages"]
        enable_search = state["enable_search"]
        info_sufficient = state["info_sufficient"]
        
        # Get the original query for context
        user_query = next((msg.content for msg in reversed(messages) 
                          if isinstance(msg, HumanMessage)), "")
        
        if info_sufficient == False and enable_search == False:
            message_content = f"""I couldn't find any relevant information about "{user_query}" in my document database.
If you enable web search, I can look for information online. 
Alternatively, you could try asking about a different topic that might be covered in the documents I have access to."""
        
        insufficient_message = AIMessage(content=message_content)
        
        return {"messages": messages + [insufficient_message]}
    
    def route_after_evaluation(self, state: AgentState) -> str: # Purely for logical routing
        """Route to the next node based on information sufficiency and search settings"""
        # If sufficient and no further action needed, go straight to final response
        if state["info_sufficient"] and not state["enable_search"]:
            return "final_response"  # Terminal
        
        # If insufficient, check if we can search the web
        if state["enable_search"]:
            return "web_search"  # Web search enabled, so use it
        else:
            return "insufficient_info"  # Terminal
    
    
    # Vector Search -> Evaluation -> Web_search(Conditional) -> Final Response [CASE: Enable_search == TRUE & INFO INSUFFICIENT]
    # Vector Search -> Evaluation -> Final Response [CASE: Enable_search flag FALSE, INFO SUFFICIENT]
    # Vector Search -> Evaluation ->  Insufficient_node [CASE: Enable_search FALSE, INFO INSUFFICIENT]
    
    def create_graph(self):
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        # workflow.add_node("assistant", self.assistant_node)
        workflow.add_node("vector_search", self.vector_search_node)
        workflow.add_node("evaluate_info", self.evaluate_info_node)
        workflow.add_node("web_search", self.web_search_node)
        workflow.add_node("final_response", self.final_response_node) # Terminal (When db search/web search deemed sufficient)
        workflow.add_node("insufficient_info", self.insufficient_info_node) # Terminal (When context search fails)
        
        # Add edges
        workflow.add_edge(START, "vector_search")
        workflow.add_edge("vector_search", "evaluate_info")
        
        # Add conditional edge from evaluate_info using the routing function
        workflow.add_conditional_edges(
        "evaluate_info",  # Source node
        self.route_after_evaluation,  # Routing function
        {
            "final_response": "final_response",  # Map return values to destination nodes
            "web_search": "web_search",
            "insufficient_info": "insufficient_info"
        }
        )
        
        workflow.add_edge("web_search", "final_response")  
        
        # Set Terminal Nodes
        workflow.add_edge("final_response", END)
        workflow.add_edge("insufficient_info", END) 
    
        
        # Compile the graph
        return workflow.compile()
    
    def invoke(self, query, enable_search=False):
        """Run the agent with the given query"""
        # Initialize state with stricter system prompt
        load_dotenv() 
        state = {
            "messages": [
                SystemMessage(content="""You are a retrieval-augmented assistant that ONLY uses information 
                provided in the context from document database or web searches.
                
                IMPORTANT:
                - If the needed information is not in the provided context, admit you don't know.
                - Do NOT use any prior knowledge that wasn't explicitly provided in the context.
                - Only reference information that appears in the search results.
                - If you're uncertain about any information, say so rather than guessing"""),
                HumanMessage(content=query)
            ],
            "info_sufficient": False,
            "enable_search": enable_search,
            "found_db_info": False
        }
        
        # Run the graph
        result = self.graph.invoke(state)
        
        # Return the last message
        return result["messages"][-1].content

# MCDONALDS HIRE ME
# client = VSPipe.setup_Qdrant_client()
# agent = RAGAgent(client=client, collection_name='Ollama-test1')
# response = agent.invoke("Why is a tomato red", enable_search=False)
# print(response)