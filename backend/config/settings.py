"""
Central configuration. Every other module reads settings from here
instead of scattering os.environ.get() calls throughout the codebase.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- LLM ---
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
    CHAT_MODEL: str = os.environ.get("CHAT_MODEL", "llama-3.3-70b-versatile")
    FAST_MODEL: str = os.environ.get("FAST_MODEL", "llama-3.1-8b-instant")  # cheap/fast for routing & simple classification

    # --- Observability ---
    LANGCHAIN_TRACING_V2: str = os.environ.get("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_API_KEY: str = os.environ.get("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.environ.get("LANGCHAIN_PROJECT", "max-ai-assistant")

    # --- Storage paths ---
    DATA_DIR: str = os.environ.get("DATA_DIR", "./data")
    CHROMA_PERSIST_DIR: str = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma_store")
    MEMORY_DB_PATH: str = os.environ.get("MEMORY_DB_PATH", "./data/memory.db")
    TRACE_DB_PATH: str = os.environ.get("TRACE_DB_PATH", "./data/traces.db")

    # --- RAG tuning ---
    CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "120"))
    RETRIEVAL_TOP_K: int = int(os.environ.get("RETRIEVAL_TOP_K", "8"))
    RERANK_TOP_N: int = int(os.environ.get("RERANK_TOP_N", "4"))
    MIN_RETRIEVAL_CONFIDENCE: float = float(os.environ.get("MIN_RETRIEVAL_CONFIDENCE", "0.35"))

    # --- Reliability ---
    LLM_TIMEOUT_SECONDS: int = int(os.environ.get("LLM_TIMEOUT_SECONDS", "30"))
    MAX_RETRIES: int = int(os.environ.get("MAX_RETRIES", "2"))

    # --- Sensitive actions requiring human approval ---
    SENSITIVE_TOOLS: set = {"send_email", "delete_memory", "delete_task"}


settings = Settings()
