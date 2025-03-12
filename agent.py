import os
import ollama
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from typing import TypedDict, List, Dict
from bert_score import score  # NEW: BERTScore for evaluation

# Define AgentState for tracking messages and info sufficiency
class AgentState(TypedDict):
    messages: List[AIMessage]
    info_sufficient: bool
    enable_search: bool
    found_db_info: bool
    bert_score: float  # NEW: Stores BERTScore evaluation

class RAGAgent:
    def __init__(self, client, collection_name, llm_model="phi4-mini"):
        self.client = client
        self.collection_name = collection_name
        self.llm_model = llm_model
        self.llm = ChatOllama(model=llm_model)
        self.graph = self.create_graph()

    def hybrid_search(self, query: str, limit: int = 5):
        """Retrieve top matching documents from Qdrant"""
        dense_vector = ollama.embeddings(model="rjmalagon/gte-qwen2-1.5b-instruct-embed-f16", prompt=query)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=models.NamedVector(name="gte-qwen1.5", vector=dense_vector['embedding']),
            limit=limit,
            search_params=models.SearchParams(hnsw_ef=128)
        )
        return results

    def vector_search_node(self, state: AgentState):
        """Retrieve relevant document chunks for RAG"""
        messages = state["messages"]
        user_query = next((msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)), "")
        results = self.hybrid_search(query=user_query, limit=5)

        context = "\n".join([result.payload.get("text", "No text found") for result in results])
        found_info = bool(context.strip())

        search_message = AIMessage(content=f"Vector Search Context:\n{context}" if found_info else "No relevant information found.")

        return {
            "messages": messages + [search_message],
            "found_db_info": found_info
        }

    def final_response_node(self, state: AgentState):
        """Generate response and evaluate with BERTScore"""
        messages = state["messages"]
        user_query = next((msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)), "")

        final_prompt = HumanMessage(content=f"Answer the question based on retrieved context:\n{messages[-1].content}")
        final_response = self.llm.invoke(messages + [final_prompt])
        
        # Compute BERTScore
        bertscore = self.evaluate_with_bertscore(final_response.content, messages[-1].content)

        return {
            "messages": messages + [final_response],
            "bert_score": bertscore  # NEW: Return BERTScore
        }

    def evaluate_with_bertscore(self, generated_text, expected_text):
        """Evaluate response similarity using BERTScore"""
        P, R, F1 = score([generated_text], [expected_text], lang="en", rescale_with_baseline=True)
        return round(F1.item(), 4)

    def create_graph(self):
        """Create the RAG agent workflow"""
        workflow = StateGraph(AgentState)
        workflow.add_node("vector_search", self.vector_search_node)
        workflow.add_node("final_response", self.final_response_node)
        workflow.add_edge(START, "vector_search")
        workflow.add_edge("vector_search", "final_response")
        workflow.add_edge("final_response", END)
        return workflow.compile()

    def invoke(self, query, enable_search=False):
        """Run the RAG agent and return response with evaluation"""
        state = {
            "messages": [SystemMessage(content="You are a retrieval-augmented assistant."), HumanMessage(content=query)],
            "info_sufficient": False,
            "enable_search": enable_search,
            "found_db_info": False,
            "bert_score": 0.0
        }
        result = self.graph.invoke(state)
        return result["messages"][-1].content, result["bert_score"]
