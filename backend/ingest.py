import os
from tqdm import tqdm

from langchain_community.document_loaders import (
    UnstructuredPDFLoader, Docx2txtLoader, CSVLoader,
    UnstructuredExcelLoader, TextLoader
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "campus-assistant")

# --- HELPER FUNCTIONS (MERGED WITH TANAY'S IMPROVEMENTS) ---

def load_single_document(file_path: str):
    """Loads a single document using the appropriate loader with Tanay's enhancements."""
    try:
        ext = os.path.splitext(file_path)[1].lower()
        # MERGED: Tanay's improved PDF loader with language support
        if ext == '.pdf':
            loader = UnstructuredPDFLoader(file_path, mode="elements", languages=['eng', 'hin'])
        elif ext == '.docx':
            loader = Docx2txtLoader(file_path)
        elif ext == '.csv':
            loader = CSVLoader(file_path, encoding="utf-8")
        elif ext == '.xlsx':
            loader = UnstructuredExcelLoader(file_path, mode="elements")
        elif ext == '.txt':
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            return None
        return loader.load()
    except Exception as e:
        print(f"Error loading {os.path.basename(file_path)}: {e}")
    return None

def chunk_documents(documents: list):
    """Chunks a list of documents using Tanay's improved splitter settings."""
    # MERGED: Tanay's improved text splitter for better context
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(documents)

# --- MAIN INGESTION FUNCTION (YOUR STRUCTURE + TANAY'S LOGIC) ---

def process_documents_for_college(college_id: str, temp_dir: str, embeddings: HuggingFaceEmbeddings):
    print(f"\n--- Processing data for college: {college_id} from path {temp_dir} ---")
    files_to_process = [f for f in os.listdir(temp_dir) if f.endswith(('.pdf', '.docx', '.csv', '.xlsx','.txt'))]

    processed = 0
    skipped = 0

    for filename in tqdm(files_to_process, desc=f"Ingesting for {college_id}"):
        file_path = os.path.join(temp_dir, filename)
        documents = load_single_document(file_path)
        if not documents:
            skipped += 1
            continue

        chunked_documents = chunk_documents(documents)
        if not chunked_documents:
            print(f"Warning: Could not create any text chunks from {filename}. Skipping.")
            skipped += 1
            continue

        for chunk in chunked_documents:
            clean_metadata = {
                "source": str(chunk.metadata.get("source", "unknown")).split('/')[-1],
                "page": str(chunk.metadata.get("page_number", "unknown"))
            }
            chunk.metadata = clean_metadata

        print(f"Embedding and uploading {len(chunked_documents)} chunks for {filename}...")
        try:
            PineconeVectorStore.from_documents(
                documents=chunked_documents,
                embedding=embeddings,
                index_name=PINECONE_INDEX_NAME,
                namespace=college_id
            )
            print(f"Successfully processed {filename}.")
            processed += 1
        except Exception as e:
            print(f"An error occurred during upload for {filename}: {e}")
            skipped += 1

    print(f"✅ Ingestion complete for '{college_id}'. Files processed: {processed}, skipped: {skipped}.")
