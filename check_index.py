import os
from dotenv import load_dotenv
from pinecone import Pinecone

# --- CONFIGURATION ---
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "campus-assistant"

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Error: Index '{PINECONE_INDEX_NAME}' does not exist.")
    else:
        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        
        print(f"--- Stats for index: '{PINECONE_INDEX_NAME}' ---")
        print(f"Total Vectors: {stats.total_vector_count}")
        print("\n--- Vectors per Namespace ---")
        if not stats.namespaces:
            print("No namespaces found.")
        else:
            for namespace, details in stats.namespaces.items():
                print(f"- {namespace}: {details.vector_count} vectors")