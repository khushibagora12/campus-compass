# app.py (root)
# Single ASGI app for Hugging Face Spaces (no uvicorn, no launch)
# - Gradio provides the HTTP server
# - We mount your FastAPI app at /api

import gradio as gr
from backend.main import app as fastapi_app   # your existing FastAPI app

# Minimal Gradio UI so the Space loads a page at "/"
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Campus Compass (Backend)
        This Space hosts the API.  
        - Health: `/api/healthz`  
        - Docs: `/api/docs`  
        - Ask endpoint: `POST /api/ask`
        """
    )

# Build a FastAPI app that serves the Gradio UI at "/"
app = gr.routes.App.create_app(demo)

# Mount your real FastAPI under /api
app.mount("/api", fastapi_app)
