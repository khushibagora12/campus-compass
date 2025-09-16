# Dockerfile
FROM python:3.11-slim
WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend application code
COPY ./backend .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]