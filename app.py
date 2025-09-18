# app.py (root)
# Hugging Face Spaces (FastAPI) expects `app` at the repo root.
# We simply re-export your existing FastAPI instance.

from backend.main import app  # exposes `app` object
