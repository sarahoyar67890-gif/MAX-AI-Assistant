"""
Observability layer.

Two tiers, both real:

1. LangSmith (optional) — if LANGCHAIN_API_KEY is set in .env, LangChain's
   native tracing kicks in automatically (env vars are read by the
   LangChain SDK itself, nothing extra to wire up). This gives you the
   full LangSmith trace UI for free once you add your key.

2. Local tracer (always on) — every graph node call is logged to a local
   SQLite DB with latency, node name, and any metadata. This works with
   zero external accounts, so `/metrics` and the frontend's "System
   Status" panel always have real data, even before you've set up
   LangSmith. This answers "why did MAX produce this answer" and
   "which part is slow" without needing an external dashboard.
"""

import sqlite3
import time
import json
import os
from contextlib import contextmanager
from datetime import datetime

from backend.config.settings import settings


def _ensure_db():
    os.makedirs(os.path.dirname(settings.TRACE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(settings.TRACE_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            node_name TEXT,
            latency_ms REAL,
            metadata TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    return conn


# NOTE: called at the start of every function below (not just once at
# import) so schema always exists against the current TRACE_DB_PATH —
# matters for tests that swap the path after this module is imported.


@contextmanager
def trace_node(node_name: str, session_id: str = "default", **metadata):
    """
    Usage:
        with trace_node("router_node", session_id=sid, route_decided="chat"):
            ... do the work ...

    Records latency automatically. Any kwargs passed in become metadata
    logged alongside the trace (e.g. which tool was called, token count).
    """
    start = time.perf_counter()
    error = None
    try:
        yield
    except Exception as e:
        error = str(e)
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        meta = dict(metadata)
        if error:
            meta["error"] = error
        _write_trace(session_id, node_name, latency_ms, meta)


def _write_trace(session_id: str, node_name: str, latency_ms: float, metadata: dict):
    conn = _ensure_db()
    conn.execute(
        "INSERT INTO traces (session_id, node_name, latency_ms, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, node_name, latency_ms, json.dumps(metadata), datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_recent_traces(limit: int = 50) -> list[dict]:
    conn = _ensure_db()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM traces ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_session_trace(session_id: str) -> list[dict]:
    """Full trace for one session — answers 'why did MAX produce this answer'."""
    conn = _ensure_db()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM traces WHERE session_id = ? ORDER BY id ASC", (session_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_node_performance_summary() -> list[dict]:
    """Which node is slow, on average — across all recorded traces."""
    conn = _ensure_db()
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT node_name,
               COUNT(*) as call_count,
               AVG(latency_ms) as avg_latency_ms,
               MAX(latency_ms) as max_latency_ms
        FROM traces
        GROUP BY node_name
        ORDER BY avg_latency_ms DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def is_langsmith_enabled() -> bool:
    return bool(settings.LANGCHAIN_API_KEY) and settings.LANGCHAIN_TRACING_V2 == "true"
