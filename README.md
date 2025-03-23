# olqRAG - Agentic RAG System

olqRAG is an open-source Retrieval-Augmented Generation (RAG) system that combines local LLMs, hybrid search, and intelligent agents to answer user queries with relevant document-based context.

This project was developed as part of AAI3008 Large Language Models at Singapore Institute of Technologu. It explores hybrid document retrieval using Qdrant, LangChain, and Ollama, allowing users to interact with locally stored knowledge efficiently.

## Features

- Document Ingestion: Uploads PDFs, text files, and Markdown.
- Hybrid Search: Combines dense embeddings, sparse retrieval, and late interaction re-ranking for accurate results.
- Agentic Reasoning: Uses LangChain's ReAct framework for step-by-step decision-making.
- Local AI Support: Runs Ollama-based LLMs (phi3:mini, mistral) for private and efficient processing.
- Google Search Integration: Optionally fetches real-time online data to augment document retrieval.
- Fast and Scalable: Uses Qdrant (vector database) for optimized similarity search.

## Installation Guide

### Clone the Repository

```bash
git clone insert finalised link
cd olqRAG
```

### Virtual Environment Setup

Set up a virtual environment on Visual Studio Code:

```bash
python -m venv llm-env
```

Activate the virtual environment:

- Windows:

```bash
llm-env\Scripts\activate
```

- Mac/Linux:

```bash
source llm-env/bin/activate
```

### Install Dependencies

Ensure you have Python 3.9+ installed, then run:

```bash
pip install -r requirements.txt
```

### Qdrant Setup

1. Create a Qdrant account at [Qdrant Tech](https://qdrant.tech/).
2. Create a cluster, and note the endpoint after creation.
3. Generate an API key and keep it accessible.

### Google Custom Search Engine (CSE) Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/).
2. Click on the menu and navigate to "APIs & Services".
3. Search for "Custom Search API" and enable it.
4. Go to "Credentials", click "Create credentials", then select "API key". Save the key.
5. Set up Google Programmable Search Engine:
   - Visit [Google Programmable Search](https://programmablesearchengine.google.com/about/).
   - Click "Get Started" and "Add" to create a new search engine.
   - Name your search engine and choose the search scope (specific sites or the entire web).
   - Click "Create" and save the search engine ID (the value after "cx=").

### Ollama Installation

1. Download Ollama from [Ollama Download](https://ollama.com/download).
2. Install and allow it to run in the background when prompted.

### Set Up API Keys

Create a `.env` file and add your API credentials:

```env
QDRANT_API_KEY=<your_qdrant_key>
QDRANT_URL=http://localhost:6333
GOOGLE_API_KEY=<your_google_search_api>
GOOGLE_CSE_API=<your_google_cse_key>
```

### Run the Application

```bash
streamlit run main.py
```

## Usage Instructions

1. **Set Up API Keys and Connections**
   - Navigate to Setup API and Connections (on the Streamlit sidebar).
   - Enter your Qdrant and Google Search API keys.
   - Click "Save API Keys".

2. **Upload Documents**
   - Go to the Upload Documents page.
   - Drag and drop a document, then click "Load and Process Document".
   - If you encounter an error, ensure the required Ollama model is installed by running:

     ```bash
     ollama pull rjmalagon/gte-qwen2-1.5b-instruct-embed-f16
     ```

     If the error persists, restart `main.py`.

3. **Ask Questions**
   - Select an Ollama Model and enter your query.
   - The system will:
     - Retrieve relevant documents from Qdrant.
     - Optionally fetch online data.
     - Generate a response using an LLM agent.

## System Architecture

1. **Document Processing**
   - Uses LangChain loaders to process files.
   - Splits text into 512-token chunks.
   - Generates embeddings using dense, sparse, and late interaction models.

2. **Vector Storage and Retrieval**
   - Stores dense, sparse, and ColBERT embeddings in Qdrant.
   - Uses hybrid search for optimal recall.
   - Performs similarity searches and re-ranking for better retrieval results.

3. **Agentic LLM Processing**
   - Implements LangChain's ReAct Agent for reasoning.
   - Uses Ollama-based models for contextualized responses.
   - Integrates search results into structured responses via the agent framework.

4. **Optional Online Search**
   - Uses Google Search API for live information retrieval.
   - Merges web search results with document-based knowledge for a hybrid response.

5. **Query Execution Flow**
   - User submits a query.
   - System retrieves document-based context.
   - Optionally fetches online search results.
   - Passes context to an LLM agent for reasoning.
   - Returns a structured response to the user.

## Technologies Used

- Programming Language: Python
- Frameworks: LangChain, Streamlit, FastAPI
- LLMs: Ollama (phi3:mini, mistral, etc.)
- Vector Database: Qdrant
- Embeddings: Dense, Sparse (BM25), Late Interaction (ColBERT)
- APIs: Google Search API

## Contributors (in alphabetical order)

- Abdul Haliq Bin Abdul Rahim
- Aldridge Melrose Tan Qi Ren
- Kuick Siqi
- Nicholas Tan Qin Sheng
- Toh Shao Ern
- Wong Khin Foong
