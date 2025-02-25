import os
from dotenv import load_dotenv

# Ollama imports
import ollama
from ollama import chat
from langchain_ollama import OllamaEmbeddings

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from langsmith import wrappers, traceable

# Embedding imports
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, models, PointStruct

class DocumentStruct:
    def __init__(self, metadata, page_content, embedding, sparse_embedding, late_interaction):
        self.metadata = metadata
        self.page_content = page_content
        self.embedding = embedding
        self.sparse_embedding = sparse_embedding
        self.late_interaction = late_interaction
        
    def __repr__(self):
        return f"DocumentEmbedding(metadata={self.metadata}, page_content={self.page_content}, embedding={self.embedding}, sparse_embedding={self.sparse_embedding}, late_interaction={self.late_interaction} )"

# Document Loader
def load_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".md"):
        loader = UnstructuredMarkdownLoader(file_path)
    documents = loader.load()
    return documents

# Chunking
@traceable
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        # Separators in order of priority (Semantic Separation)
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
        keep_separator=True,
        strip_whitespace=True
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Dense Vectors Generation
def generate_dense_vector(text: str):
    dense_model_path = 'rjmalagon/gte-qwen2-1.5b-instruct-embed-f16'
    dense_model = OllamaEmbeddings(model=dense_model_path)
    dense_vector = dense_model.embed_query(text)
    # print(len(dense_vector)) # Show Dims
    
    return dense_vector

@traceable
def generateEmbeddings(documents):
    embeddingsList = []
    late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
    sparse_embeddings_model = SparseTextEmbedding("Qdrant/bm25")
    dense_model = OllamaEmbeddings(model='rjmalagon/gte-qwen2-1.5b-instruct-embed-f16')

    for idx, item in enumerate(documents):
        metadata = item.metadata
        page_content = item.page_content
        embedding = dense_model.embed_query(item.page_content)
        sparse_embedding = list(sparse_embeddings_model.embed(item.page_content))
        late_interaction_embed = list(late_interaction_embedding_model.embed(item.page_content))
        
        # Each item will have a sparse and dense vector embedding. Which will be then used for hybrid search functions.   
        embeddingsList.append(DocumentStruct(metadata=metadata, 
                                             page_content=page_content, 
                                             embedding=embedding, 
                                             sparse_embedding=sparse_embedding,
                                             late_interaction=late_interaction_embed
                                             )
                              )
        
    return embeddingsList


def setup_Qdrant_client():
    client = QdrantClient(
        url=os.environ["QDRANT_URL"], 
        api_key=os.environ["QDRANT_API_KEY"],
    )
    return client

def create_collection(client, collection_name):
    client.create_collection(
        collection_name,
        vectors_config={
            "gte-qwen1.5": models.VectorParams(
                size=1536,
                distance=models.Distance.COSINE,
            ),
            "colbertv2.0": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                )
            ),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF
            )
        }
    )
    print(f"Created new collection: {collection_name}")
    return collection_name

def prepare_vector_points(embeddingsList):
    points = []
    
    for idx, item in enumerate(embeddingsList):
        point = PointStruct(
            id = idx,
            vector = {
                "gte-qwen1.5": item.embedding,
                "bm25": models.SparseVector(indices=item.sparse_embedding[0].indices, values=item.sparse_embedding[0].values),
                "colbertv2.0": item.late_interaction[0],
            },
            payload={ # Payload: Provide context for points
                    "text": item.page_content,
                    "metadata": {
                        "source": item.metadata['source'],
                        "page": item.metadata['page'],
                    }
                },
        )
        points.append(point)
    return points

def upsert_points(client, collection_name, points, batch_size=10):
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        try:
            print(f"Upserting batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}")
            client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True 
            )
        except Exception as e:
            print(f"Error with batch {i//batch_size + 1}: {str(e)}")

# def main():
#     # Load environment variables
#     load_dotenv()
    
#     # Specify the file path to process
#     file_path = 'C:/Users/shaoe/OneDrive/Desktop/AAI3008LLM/The_Hundred_page_Machine_Learning_Book_Andriy_Burkov_Z_Library.pdf'
    
#     # Load and process the document
#     print(f"Loading document: {file_path}")
#     documents = load_document(file_path)
    
#     print(f"Creating chunks from document")
#     chunks = create_chunks(documents)
#     print(f"Created {len(chunks)} chunks")
    
#     print("Generating embeddings (this may take some time)...")
#     embeddings_list = generateEmbeddings(chunks)
#     print(f"Generated embeddings for {len(embeddings_list)} chunks")
    
#     # Set up Qdrant client and collection
#     client = setup_Qdrant_client()
#     client.delete_collection("Ollama-RAG")
#     collection_name = create_collection(client)
    
#     # Prepare and upsert vectors
#     points = prepare_vector_points(embeddings_list)
#     upsert_points(client, collection_name, points)
    
#     print("Document processing and vector storage complete!")

# if __name__ == "__main__":
#     main()




