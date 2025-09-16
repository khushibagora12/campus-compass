# 🧭 Campus Compass

An AI-powered, multilingual chatbot for navigating college information.

---
### ## Table of Contents
1.  [The Problem](#the-problem-)
2.  [Our Solution](#our-solution-)
3.  [Tech Stack](#tech-stack-)
4.  [Project Status](#project-status-%EF%B8%8F)
5.  [Team Roles & Workflow](#team-roles--workflow-)
6.  [Final Integration Plan](#final-integration-plan-)

---
### #The Problem 😥
College administrative offices are overwhelmed with hundreds of repetitive student queries on topics like fee deadlines, forms, and schedules. This creates long queues and strains staff. Furthermore, many students are more comfortable in regional languages, leading to communication gaps.

---
### #Our Solution ✨
**Campus Compass** is a sophisticated, multilingual AI assistant designed to solve this problem. It provides instant, 24/7, and accurate answers to student questions in their native language. Our solution ingests all official college documents (PDFs, DOCs, etc.) to ensure its responses are always correct and context-aware.

---
### Tech Stack 🛠️
* **Backend:** Python, FastAPI, LangChain
* **Frontend:** Next.js (React)
* **AI Model:** `paraphrase-multilingual-mpnet-base-v2` (for embeddings) & Google's Gemini API (for generation)
* **Database:** Pinecone (Vector DB)
* **Deployment:** Docker, Render (Backend), Vercel (Frontend)

---
### Project Status 🗓️

#### **✅ Completed**
* **Core AI Engine (by Tanay):**
    * A sophisticated data ingestion pipeline (`ingest.py`) that can read multiple file formats (PDFs, DOCX, CSVs), handle scanned documents (OCR), and understands multiple Indian languages.
    * The system is multi-tenant, ready to support multiple colleges using secure `namespaces`.
* **Core Backend API (by Tanay):**
    * A functional local server (`main.py`) powered by our RAG engine.
    * The API supports conversational memory (follow-up questions) and has an intelligent human fallback mechanism.

#### **⏳ Remaining**
The core AI is built. The remaining work involves making it production-ready, building the user interfaces, and preparing our final presentation.

---
### Team Roles & Workflow 📋

This is the official plan. Please stick to these tasks to ensure a smooth workflow.

### **🧠 AI/ML - Tanay**
Your role is to support the other teams and refine the core engine.
* **Task 1: Support Backend:** Help Shivam & Shatakshi understand the `ingest.py` and `main.py` scripts and advise on the best way to trigger the ingestion process from the Admin Dashboard.
* **Task 2: Support Frontend:** Clearly explain the API's JSON payload requirements (`query`, `college_id`, `session_id`) to Khushi & Sahil so they can connect the UI correctly.
* **Task 3 (Stretch Goal):** If time permits, experiment with prompt engineering to further improve the quality of the AI's answers.

---
### **⚙️ Backend Team - Shivam & Shatakshi**
Your goal is to turn the local Python scripts into a robust, deployed, and manageable online service.

* **Flow:**
    1.  **Finalize the API:** Clean up `main.py`, add comments, and implement robust error handling.
    2.  **Containerize with Docker:** Create a `Dockerfile` for the entire backend application.
    3.  **Deploy v1:** Deploy the containerized API to **Render**. **Your first priority is to get a live API URL to the Frontend team.**
    4.  **Build Admin Dashboard:** While the frontend team works, build the secure web interface for college admins with:
        * Admin login.
        * A feature to add new colleges (creating new `namespaces`).
        * The drag-and-drop file uploader.
        * A button/function to trigger the `ingest.py` script for newly uploaded files.
    5.  **Deploy v2:** Update the deployment on Render with the finished Admin Dashboard.
* **Deliverable:** A stable, live API URL and a functional Admin Dashboard.

---
### **🎨 Frontend Team - Khushi & Sahil**
Your goal is to build the beautiful and intuitive chat interface that students will use.

* **Flow:**
    1.  **Design the UI:** Quickly design a clean, simple, and mobile-friendly chat widget.
    2.  **Build the Component:** Develop the chat widget using **Next.js**. Focus on managing the conversation history state.
    3.  **Connect to the Live API:** Once the backend team provides the live URL, integrate your widget to make `POST` requests to their `/api/ask` endpoint.
        * Ensure you are sending `query`, `college_id`, and `session_id` in your requests.
        * Handle and display the JSON response from the backend.
        * Implement a special UI for the `fallback: true` message.
    .   **Deploy:** Deploy the finished Next.js application to **Vercel**.
* **Deliverable:** A live URL for the user-facing chat widget.

---
### **📝 Presentation & Documentation - Parth**
Your goal is to craft a compelling story and presentation that will win the hackathon.

* **Flow:**
    1.  **Content & Script:** Immediately start writing the presentation script. Clearly articulate the **Problem**, our innovative **Solution (RAG)**, the **Tech Stack**, and the **Business Value**.
    2.  **Slide Deck Design:** Create a visually stunning presentation in Canva, Figma, or Google Slides. Get mockups and logos from Tanay or generate them with AI.
    3.  **Demo Plan:** Work with the Frontend and Backend teams to script a smooth, end-to-end live demo. Plan for potential failures and have backup screenshots/videos.
    4.  **Update this README:** Keep this `README.md` file updated. Add a "How to Run Locally" section with instructions for setting up the project.
* **Deliverable:** A polished slide deck and a well-rehearsed demo plan.

---
### Final Integration Plan 🤝
1.  The **Backend Team** deploys the core API and gives the URL to the **Frontend Team**.
2.  The **Frontend Team** integrates the API and deploys the chat widget.
3.  The **Final Step:** We will take the deployed frontend widget's `<script>` tag and embed it into a simple sample college webpage to demonstrate a complete, end-to-end working product for the final demo.