"""
MAX AI Assistant — FastAPI application entrypoint.

Run locally:
    uvicorn main:app --reload --port 8000

Run with Docker:
    docker compose up --build
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router

app = FastAPI(
    title="MAX AI Assistant API",
    description="Personal AI Operations Assistant — multi-agent, RAG, memory, tools, evaluation, observability.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend's real origin before shipping
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {"status": "MAX is online", "docs": "/docs"}
