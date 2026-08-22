# MAX AI Assistant — Backend Dockerfile
FROM python:3.12-slim

WORKDIR /app

# System deps needed by sentence-transformers/chromadb (minimal, not full CUDA stack)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY main.py .
COPY data/sample_docs/ ./data/sample_docs/

# Data directory for SQLite DBs + Chroma persistence (mount as a volume in
# docker-compose so data survives container restarts)
RUN mkdir -p /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
