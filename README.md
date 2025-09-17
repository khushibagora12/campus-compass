# 🧭 Campus Compass
An AI-powered, multilingual chatbot for navigating college information.

---

## Table of Contents
1. [The Problem](#the-problem-)
2. [Our Solution](#our-solution-)
3. [Tech Stack](#tech-stack-)
4. [Project Status](#project-status-)
5. [Team Roles & Workflow](#team-roles--workflow-)
6. [Final Integration Plan](#final-integration-plan-)
7. [Development Setup](#development-setup-)
8. [Versioning & Constraints](#versioning--constraints-)
9. [Changelog](#changelog-)

---

## The Problem 😥
College administrative offices are overwhelmed with hundreds of repetitive student queries on topics like fee deadlines, forms, and schedules. This creates long queues and strains staff. Furthermore, many students are more comfortable in regional languages, leading to communication gaps.

---

## Our Solution ✨
**Campus Compass** is a sophisticated, multilingual AI assistant designed to solve this problem. It provides instant, 24/7, and accurate answers to student questions in their native language. Our solution ingests all official college documents (PDFs, DOCs, etc.) to ensure its responses are always correct and context-aware.

---

## Tech Stack 🛠️
- **Backend:** Python, FastAPI, LangChain  
- **Frontend:** Next.js (React)  
- **AI Model:** `paraphrase-multilingual-mpnet-base-v2` (embeddings) & Google Gemini API (generation)  
- **Database:** Pinecone (Vector DB)  
- **Cache & Memory:** Redis (Render)  
- **Deployment:** Docker, Render (Backend), Vercel (Frontend)  

---

## Project Status 🗓️

### ✅ Completed
- **Core AI Engine (by Tanay):**
  - Multi-format ingestion pipeline (`ingest.py`) with OCR + multilingual support.
  - Multi-tenant support via namespaces.
- **Core Backend API (by Tanay):**
  - RAG engine + conversational memory with Redis.
  - Human fallback mechanism for missing answers.

### ⏳ Remaining
- Containerization + deployment to Render.  
- Admin Dashboard (file upload + ingestion trigger).  
- Student-facing chat widget (Next.js → Vercel).  
- Final slide deck + polished demo.  

---

## Team Roles & Workflow 📋

### 🧠 AI/ML - Tanay
Support backend & frontend teams, refine core engine.  
- Help backend understand ingestion & RAG pipeline.  
- Help frontend with API payloads (`query`, `college_id`, `session_id`).  
- Optional: experiment with prompt engineering for better answers.  

### ⚙️ Backend Team - Shivam & Shatakshi
Turn scripts into a deployed, robust service.  
- Finalize `main.py`, containerize with Docker.  
- Deploy first version to Render (API URL).  
- Build Admin Dashboard (login, file upload, ingestion trigger).  
- Deploy updated backend with dashboard.  

### 🎨 Frontend Team - Khushi & Sahil
Build the student-facing chat widget.  
- Design a clean, mobile-friendly UI.  
- Develop in Next.js with conversation history.  
- Connect to backend API (`/api/ask`).  
- Handle `fallback: true` messages gracefully.  
- Deploy to Vercel.  

### 📝 Presentation & Documentation - Parth
Craft the narrative and presentation.  
- Write the pitch script (Problem → Solution → Tech → Business).  
- Build slides with visuals/mockups.  
- Prepare live demo with backups.  
- Keep this README updated.  

---

## Final Integration Plan 🤝
1. Backend Team deploys API → shares URL with Frontend Team.  
2. Frontend Team integrates API and deploys chat widget.  
3. For demo: embed the deployed widget’s `<script>` tag into a sample college webpage.  

---

## Development Setup ⚙️

### 1. Clone the repo & create venv
```bash
git clone <repo-url>
cd Campus-Compass
python -m venv .venv
.venv\Scripts\activate
````

### 2. Install dependencies

We use both a **spec file** (`requirements.txt`) and a **constraints file** (`constraints.txt`) for reproducibility.

```bash
pip install -r requirements.txt -c constraints.txt
```

If you just want to run locally (not strict):

```bash
pip install -r requirements.txt
```

### 3. Run backend locally

```bash
uvicorn backend.main:app --reload
```

Docs available at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 4. Environment variables

Create a `.env` with:

```
PINECONE_API_KEY=xxxx
GOOGLE_API_KEY=xxxx
PINECONE_INDEX_NAME=campus-assistant
REDIS_URL=rediss://<your-tls-url-from-render>
STORAGE_BUCKET_NAME=college-documents
```

---

## Versioning & Constraints 📌

Pinned critical packages for stability:

```txt
langchain==0.2.15
langchain-community==0.2.15
langchain-pinecone==0.1.2
pinecone[grpc]==5.1.0
```

* We **do not pin** `langchain-core`; pip resolves a compatible 0.2.x (currently `0.2.43`).
* A `constraints.txt` file is used to freeze exact versions from a working environment.

### Example `constraints.txt`

```txt
langchain==0.2.15
langchain-community==0.2.15
langchain-core==0.2.43
langchain-pinecone==0.1.2
pinecone==5.1.0
fastapi==0.110.0
uvicorn==0.29.0
python-dotenv==1.0.1
jinja2==3.1.4
python-multipart==0.0.9
langchain-google-genai==1.0.4
langchain-huggingface==0.0.3
redis==5.0.4
supabase==2.4.2
httpx==0.27.0
passlib==1.7.4
unstructured==0.18.15
# …plus all transitive deps frozen by pip freeze
```

To regenerate:

```bash
pip freeze > constraints.txt
```

---

## Changelog 📝

### 2025-09-17

* Fixed dependency resolution conflicts in LangChain stack:

  * Aligned on `langchain==0.2.15`, `langchain-community==0.2.15`, `langchain-pinecone==0.1.2`.
  * Adopted `pinecone[grpc]==5.1.0` (v5 SDK) to support `from pinecone import Pinecone`.
  * Removed `langchain-redis` (conflicted with core 0.2.x, not needed since we use `RedisChatMessageHistory` from `langchain_community`).
* Updated README with development setup, environment variables, and versioning strategy.
* Confirmed environment stability with test queries.


