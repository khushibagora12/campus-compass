"""
This script handles the ingestion of various document types (PDF, DOCX, CSV, XLSX)
for multiple colleges into a Pinecone vector database. It includes OCR support
and metadata cleaning for compatibility.
"""

import os
import time
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    TextLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pinecone import Pinecone, ServerlessSpec

# --- CONFIGURATION ---
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "campus-assistant"
SOURCE_DATA_PATH = "data/"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LOG_FILE_NAME = "_processed_files.log"

# --- HELPER FUNCTIONS ---

def get_file_identifier(file_path):
    """Creates a unique identifier for a file based on its path, size, and last modification time."""
    size = os.path.getsize(file_path)
    mod_time = os.path.getmtime(file_path)
    return f"{file_path}|{size}|{mod_time}"

def load_processed_log(log_path):
    """Loads the set of processed file identifiers from the log file."""
    if not os.path.exists(log_path):
        return set()
    with open(log_path, "r") as f:
        return set(f.read().splitlines())

def update_processed_log(log_path, identifier):
    """Adds a new file identifier to the log file."""
    with open(log_path, "a") as f:
        f.write(identifier + "\n")

def load_single_document(file_path):
    """Loads a single document using the appropriate loader."""
    try:
        if file_path.endswith('.pdf'):
            loader = UnstructuredPDFLoader(file_path, mode="elements")
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith('.csv'):
            loader = CSVLoader(file_path, encoding="utf-8")
        elif file_path.endswith('.xlsx'):
            loader = UnstructuredExcelLoader(file_path, mode="elements")
        elif file_path.endswith('.txt'): # <-- Add this block
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            return None
        return loader.load()
    except Exception as e:
        print(f"Error loading {os.path.basename(file_path)}: {e}")
    return None

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Chunks a list of documents."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

def get_embeddings():
    """Initializes the embedding model."""
    print("Initializing embedding model...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})

# --- MAIN INGESTION FUNCTION ---

def ingest_data_for_college(college_id, college_path, index, embeddings):
    """Processes only new or updated documents for a single college."""
    print(f"\n--- Processing data for college: {college_id} ---")
    log_path = os.path.join(college_path, LOG_FILE_NAME)
    processed_files_log = load_processed_log(log_path)
    
    files_to_process = [f for f in os.listdir(college_path) if f.endswith(('.pdf', '.docx', '.csv', '.xlsx','.txt'))]
    
    for filename in tqdm(files_to_process, desc=f"Ingesting for {college_id}"):
        file_path = os.path.join(college_path, filename)
        file_identifier = get_file_identifier(file_path)
        
        if file_identifier in processed_files_log:
            continue

        print(f"\nNew/updated file detected: {filename}")
        
        documents = load_single_document(file_path)
        if not documents:
            continue
            
        chunked_documents = chunk_documents(documents)
        raw_text_output_path = f"{filename}.txt"
        with open(raw_text_output_path, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(doc.page_content + "\n\n")
        print(f"==> Raw extracted text saved to: {raw_text_output_path}")
        if not chunked_documents:
            print(f"Warning: Could not extract any text chunks from {filename}. Skipping.")
            continue
        
        # --- METADATA CLEANING FIX ---
        # We create a new, clean metadata dictionary for each chunk,
        # keeping only the 'source' and 'page' information.
        for chunk in chunked_documents:
            clean_metadata = {
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page_number", "unknown")
            }
            chunk.metadata = clean_metadata

        print(f"Embedding and uploading {len(chunked_documents)} chunks for {filename} to namespace '{college_id}'...")
        
        try:
            vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings, namespace=college_id)
            vectorstore.add_documents(chunked_documents)
            
            update_processed_log(log_path, file_identifier)
            print(f"Successfully processed and logged {filename}.")

        except Exception as e:
            print(f"!!!!!!!!!!!!!! AN ERROR OCCURRED DURING UPLOAD !!!!!!!!!!!!!!")
            print(f"Failed to upload {filename} for college {college_id}.")
            print(f"Error details: {e}")

# --- SCRIPT EXECUTION ---

if __name__ == "__main__":
    print("Starting data ingestion process...")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print("Waiting for index to initialize...")
        time.sleep(10)
    
    index = pc.Index(PINECONE_INDEX_NAME)
    
    embeddings = get_embeddings()

    college_folders = [d for d in os.listdir(SOURCE_DATA_PATH) if os.path.isdir(os.path.join(SOURCE_DATA_PATH, d))]
    
    if not college_folders:
        print(f"Error: No college sub-directories found in '{SOURCE_DATA_PATH}'.")
    else:
        for college_id in college_folders:
            college_path = os.path.join(SOURCE_DATA_PATH, college_id)
            ingest_data_for_college(college_id, college_path, index, embeddings)
            
    print("\nData ingestion process finished.")