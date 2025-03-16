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
    def __init__(self, client, collection_name, llm_model="phi3:mini", 
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
        info_sufficient = state["info_sufficient"]
        enable_search = state["enable_search"]
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
            "found_db_info": found_info,
            "info_sufficient": info_sufficient,
            "enable_search": enable_search
        }
   
    
    def evaluate_info_node(self, state: AgentState):
        """Evaluate if information is sufficient to answer the query"""
        messages = state["messages"]
        found_db_info = state["found_db_info"]
        enable_search = state["enable_search"]
        
        # If we didn't find any info in the database, don't even need to evaluate
        if not found_db_info:
            return {"info_sufficient": False, 
                    "messages": messages,
                    "enable_search": enable_search,
                    "found_db_info": found_db_info
                    
                    }
        
        # Get the last user query
        user_query = next((msg.content for msg in reversed(messages) 
                        if isinstance(msg, HumanMessage)), "")
        
        # Get the vector search context
        vector_content = next((msg.content for msg in messages 
                            if isinstance(msg, AIMessage) and "Vector Search Context:" in msg.content), "")
        
        # Ask LLM to evaluate if the information is sufficient - use a clearer prompt
        eval_prompt = HumanMessage(content=f"""
        I need you to evaluate if the following Vector Search Context has SUFFICIENT information to answer this query:
        
        User Query: "{user_query}"
        
        Context: {vector_content}
        
          IMPORTANT INSTRUCTIONS:
        - If the context contains ANY definition, explanation, or relevant information about the query, output ONLY: "SUFFICIENT"
        - If the context contains ABSOLUTELY NOTHING related to the query, output ONLY: "INSUFFICIENT"

          Include ONLY one word as the output: SUFFICIENT or INSUFFICIENT
              """)
        
        # Log the evaluation prompt for debugging
        print(f"Evaluating sufficiency with query: {user_query}")
        
        # Get the evaluation from the LLM
        eval_response = self.llm.invoke(input=messages[-1:] + [eval_prompt])
        eval_text = eval_response.content.strip().upper()
        
        # Log the raw LLM response for debugging
        print(f"Raw LLM evaluation response: {eval_text}")
        
        # Determine sufficiency from response
        is_sufficient = (eval_text == "SUFFICIENT")
        
        # Log the final decision
        print(f"Final sufficiency determination: {is_sufficient}")
        
        return {
        "info_sufficient": is_sufficient, 
        "messages": messages,
        "enable_search": enable_search,
        "found_db_info": found_db_info
        }
    
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

        user_query = next((msg.content for msg in reversed(messages) 
                      if isinstance(msg, HumanMessage)), "")
        
        final_prompt = HumanMessage(content=f"""
        Please answer this question concisely and clearly: "{user_query}"
    
        Follow these guidelines:
        1. Use ONLY information from the search results - do not add any external knowledge
        2. Write in a clear, straightforward style with no unnecessary jargon
        3. Keep your answer focused and to the point
        4. If the search results don't contain enough information on a specific aspect, simply state "Based on the available information, I cannot determine [specific aspect]"
        
        Your answer should be structured like this:
        
        ANSWER: [Your direct answer to the question using information from the searches]
        
        SOURCES:
        - Database: [Brief mention of what information came from the database]
        - Web Search: [Brief mention of what information came from web search, omit if none]
        """)
        
        final_response = self.llm.invoke(input=[
        SystemMessage(content="""You are a precise, clear assistant that provides direct answers based only on provided information.
        Be concise and straightforward. Avoid unnecessary words, repetition, and jargon.
        Never include instructions or meta-commentary in your responses."""),
        final_prompt
        ])

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
        enable_search = state["enable_search"]
        info_sufficient = state["info_sufficient"]

        if info_sufficient and not enable_search:
            return "final_response"  # Terminal
        
        if enable_search:
            return "web_search"
        else:
            return "insufficient_info"
    
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