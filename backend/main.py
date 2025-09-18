import os
import uuid
import shutil
import time
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request, Form, File, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.schema.runnable import RunnableLambda
from pinecone import Pinecone

from backend.supabase_client import supabase, supabase_admin
from backend.auth_utils import verify_password
from backend.ingest import process_documents_for_college

# --- CONFIGURATION ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "campus-assistant")
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL_NAME = "gemini-1.5-flash"
REDIS_URL = os.getenv("REDIS_URL")
STORAGE_BUCKET_NAME = os.getenv("STORAGE_BUCKET_NAME", "college-documents")

# --- APP SETUP ---
app = FastAPI(title="Campus Compass API", version="5.0.0 Final")
templates = Jinja2Templates(directory="backend/templates")

# Globals filled during startup()
pc = None
index = None
embeddings = None
llm = None
READY = False
INIT_ERROR = None

# --- PROMPT SETUP ---
prompt_template = """
You are a helpful assistant for college students named Campus Compass. Provide accurate answers based on the context and chat history.
If the context does not contain the answer, state "I_DO_NOT_KNOW" and nothing else.

CHAT HISTORY:
{chat_history}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
prompt = PromptTemplate.from_template(prompt_template)

# --- DATA MODELS ---
class QueryRequest(BaseModel):
    college_id: str
    query: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

# --- STARTUP: defer heavy init so the port opens quickly ---
@app.on_event("startup")
def startup_init():
    global pc, index, embeddings, llm, READY, INIT_ERROR
    print("Initializing services (deferred)...", flush=True)
    try:
        # Initialize clients
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)

        # HuggingFaceEmbeddings will download model on first use if missing
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )

        # Google LLM client
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0.3)

        READY = True
        print("Services initialized.", flush=True)
    except Exception as e:
        INIT_ERROR = str(e)
        READY = False
        # Do not crash the process—allow health endpoint to report the issue
        print(f"Startup init failed: {INIT_ERROR}", flush=True)

# --- HEALTH/READINESS ---
@app.get("/healthz")
def health():
    if INIT_ERROR:
        return JSONResponse({"ok": False, "error": INIT_ERROR}, status_code=500)
    return {"ok": READY}

# --- STUDENT API ENDPOINT ---
@app.post("/api/ask", response_model=dict)
async def ask_question(request: QueryRequest):
    if not READY:
        msg = "Warming up the AI engine (downloading models). Please retry in ~30–60 seconds."
        if INIT_ERROR:
            msg = f"Service init error: {INIT_ERROR}"
        return {"answer": msg, "session_id": request.session_id, "fallback": True}

    print(f"Session '{request.session_id}' | Query for '{request.college_id}': {request.query}", flush=True)

    try:
        memory = RedisChatMessageHistory(session_id=request.session_id, url=REDIS_URL)

        query_vector = embeddings.embed_query(request.query)
        query_results = index.query(
            namespace=request.college_id, vector=query_vector, top_k=3, include_metadata=True
        )
        context = "\n\n".join([match.metadata.get('text', '') for match in query_results.matches if match.metadata])

        rag_chain = prompt | llm | StrOutputParser()
        chat_history_messages = memory.messages
        answer = rag_chain.invoke({
            "context": context,
            "question": request.query,
            "chat_history": chat_history_messages
        })

        memory.add_user_message(request.query)
        memory.add_ai_message(answer)

        if "I_DO_NOT_KNOW" in answer or not context.strip():
            fallback_message = "I couldn't find a specific answer in my documents. For assistance, please contact the administration office."
            return {"answer": fallback_message, "session_id": request.session_id, "fallback": True}
        else:
            return {"answer": answer, "session_id": request.session_id, "fallback": False}

    except Exception as e:
        print(f"An error occurred in /api/ask: {e}", flush=True)
        return {"error": "Failed to generate an answer.", "details": str(e)}

# --- ADMIN DASHBOARD (unchanged) ---
@app.get("/admin", response_class=HTMLResponse)
async def get_admin_login(request: Request, error: str = None):
    error_message = "Invalid username or password." if error else None
    return templates.TemplateResponse("login.html", {"request": request, "error_message": error_message})

@app.post("/admin/login", response_class=HTMLResponse)
async def post_admin_login(request: Request, username: str = Form(...), password: str = Form(...)):
    clean_username = username.strip()
    response = supabase.table('admins').select("*").eq('username', clean_username).execute()

    if response.data:
        user = response.data[0]
        if verify_password(password.strip(), user['hashed_password']):
            file_list = supabase_admin.storage.from_(STORAGE_BUCKET_NAME).list(user['college_id'])
            return templates.TemplateResponse("dashboard.html", {
                "request": request,
                "username": user['username'],
                "college_id": user['college_id'],
                "files": file_list
            })
    return RedirectResponse(url="/admin?error=true", status_code=303)

@app.post("/admin/upload")
async def upload_documents(college_id: str = Form(...), files: List[UploadFile] = File(...)):
    for file in files:
        file_path_in_bucket = f"{college_id}/{file.filename}"
        file_content = await file.read()
        try:
            supabase_admin.storage.from_(STORAGE_BUCKET_NAME).upload(
                path=file_path_in_bucket,
                file=file_content,
                file_options={"content-type": file.content_type, "x-upsert": "true"}
            )
        except Exception as e:
            return {"error": f"Failed to upload {file.filename}: {str(e)}"}

    return {"message": f"{len(files)} files uploaded successfully to '{college_id}'."}

@app.post("/admin/ingest")
async def trigger_ingestion(background_tasks: BackgroundTasks, college_id: str = Form(...)):
    temp_dir = f"temp_{college_id}_{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        files_in_bucket = supabase_admin.storage.from_(STORAGE_BUCKET_NAME).list(college_id)
        for file_obj in files_in_bucket:
            file_name = file_obj['name']
            file_path_in_bucket = f"{college_id}/{file_name}"
            local_file_path = os.path.join(temp_dir, file_name)

            with open(local_file_path, 'wb+') as f:
                res = supabase_admin.storage.from_(STORAGE_BUCKET_NAME).download(file_path_in_bucket)
                f.write(res)

        print(f"Adding ingestion for college '{college_id}' to background tasks...", flush=True)
        background_tasks.add_task(process_documents_for_college, college_id, temp_dir, embeddings)
        background_tasks.add_task(time.sleep, 300)
        background_tasks.add_task(shutil.rmtree, temp_dir)

    except Exception as e:
        shutil.rmtree(temp_dir)
        return {"error": f"Failed to prepare files for ingestion: {str(e)}"}

    return {"message": f"Ingestion process for '{college_id}' has been started."}

# --- ROOT ENDPOINT ---
@app.get("/")
def read_root():
    return {"message": "Campus Compass API is running."}
