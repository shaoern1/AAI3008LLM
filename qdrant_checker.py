from qdrant_client import QdrantClient

# Your Qdrant API credentials
QDRANT_API_KEY = "lEAydiEwlkTw0KDF_exJI8PqTgGWCIYznzY1y3u6eRxs0cPGaeODcQ"
QDRANT_URL = "https://52f54627-3f27-403e-a9e4-561e4d8949d0.europe-west3-0.gcp.cloud.qdrant.io:6333"

# Connect to Qdrant Cloud
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# List all available collections
collections = client.get_collections()
print("\n✅ Available Collections in Qdrant:")
print(collections)

# Replace with your actual collection name
COLLECTION_NAME = "aldotest"  

# Check if the collection exists
if COLLECTION_NAME in [c.name for c in collections.collections]:
    print(f"\n✅ Collection '{COLLECTION_NAME}' exists!")

    # Check the number of stored vectors
    count = client.count(COLLECTION_NAME)
    print(f"\n📌 Number of stored embeddings in '{COLLECTION_NAME}': {count.count}")

    # Fetch a few sample records
    results = client.scroll(COLLECTION_NAME, limit=5)
    print("\n📜 Sample stored documents:")
    for point in results[0]:
        print(f"📝 Text: {point.payload.get('text', 'No text found')}\n")

else:
    print(f"\n❌ Collection '{COLLECTION_NAME}' NOT FOUND in Qdrant. Check your database or re-upload documents.")
