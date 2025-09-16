"""
Backend server for the Campus Assistant chatbot "Polyglot".
This FastAPI application provides an endpoint to answer student questions using a 
RAG pipeline with conversational memory and a human fallback mechanism.
"""
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import uuid

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory

# --- CONFIGURATION ---
load_dotenv()

# Use your new index name if you changed it, e.g., "polyglot-assistant"
PINECONE_INDEX_NAME = "campus-assistant" 
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL_NAME = "gemini-1.5-flash"

# --- DATA MODELS ---

class QueryRequest(BaseModel):
    college_id: str
    query: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

# --- INITIALIZATION ---

app = FastAPI(
    title="Polyglot API",
    description="API for the Polyglot campus assistant with memory and human fallback.",
    version="1.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME, 
    model_kwargs={'device': 'cpu'}
)
print("Embedding model initialized.")

llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0.3)

# A simple in-memory store for chat histories. In production, use Redis or a database.
chat_histories = {}

# --- PROMPT & CHAIN SETUP ---

prompt_template = """
You are a helpful assistant for college students named Polyglot. Your goal is to provide accurate and concise answers based only on the context provided.
Use the chat history to understand the context of follow-up questions.
If the context does not contain the information needed to answer the question, you must respond with the exact phrase "I_DO_NOT_KNOW" and nothing else.

CHAT HISTORY:
{chat_history}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history", "context", "question"])

# --- API ENDPOINT ---

@app.post("/api/ask", response_model=dict)
async def ask_question(request: QueryRequest):
    """
    Receives a question, retrieves context, considers chat history, and generates an answer.
    If the answer is not found, it returns a human fallback response.
    """
    print(f"Session '{request.session_id}' | Query for '{request.college_id}': {request.query}")

    # Get or create the memory for this session
    if request.session_id not in chat_histories:
        chat_histories[request.session_id] = ConversationBufferMemory()
    memory = chat_histories[request.session_id]

    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME, 
        embedding=embeddings, 
        namespace=request.college_id
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3}) # Retrieve top 3 chunks
    print("\n--- DEBUG: Retrieving documents for query ---")
    retrieved_docs = retriever.invoke(request.query)
    print("--- RETRIEVED DOCUMENTS ---")
    for i, doc in enumerate(retrieved_docs):
        print(f"--- Document {i+1} ---")
        print(doc.page_content)
        print("--------------------")
    # Define the RAG chain
    rag_chain = (
        {
            "context": retriever, 
            "question": RunnablePassthrough(),
            "chat_history": RunnableLambda(lambda x: memory.load_memory_variables(x)["history"])
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    try:
        answer = rag_chain.invoke(request.query)
        
        # Save the context to memory
        memory.save_context({"input": request.query}, {"output": answer})

        # --- HUMAN FALLBACK LOGIC ---
        if "I_DO_NOT_KNOW" in answer:
            print(f"Session '{request.session_id}' | Answer not found. Triggering human fallback.")
            fallback_message = "I couldn't find a specific answer in my documents. For assistance, please contact the administration office."
            return {"answer": fallback_message, "session_id": request.session_id, "fallback": True}
        else:
            print(f"Session '{request.session_id}' | Answer: {answer}")
            return {"answer": answer, "session_id": request.session_id, "fallback": False}

    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": "Failed to generate an answer.", "details": str(e)}

@app.get("/")
def read_root():
    return {"message": "Polyglot API is running."}

# --- To run this server, use the command in your terminal: ---
# uvicorn main:app --reload