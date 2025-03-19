import redis
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Connect to Redis
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"), 
    port=int(os.getenv("REDIS_PORT", 6379)), 
    db=0, decode_responses=True
)

def get_chat_history(session_id):
    """Fetch chat history from Redis"""
    history = redis_client.get(f"chat_history:{session_id}")
    return json.loads(history) if history else []

def save_chat_history(session_id, messages):
    """Save chat history to Redis"""
    redis_client.set(f"chat_history:{session_id}", json.dumps(messages))
