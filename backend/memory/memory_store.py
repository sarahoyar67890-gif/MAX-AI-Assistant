"""
Memory system.

Three deliberately separate stores, as specified:

    short_term  -> current conversation only, kept in-process, trimmed
                   to the last N turns. Never persisted to disk.

    long_term   -> facts worth remembering across sessions. NOT written
                   automatically for everything — only written when
                   explicitly flagged as memory-worthy (see
                   `should_persist` in memory_agent.py). User can list
                   and delete anything stored here.

    preferences -> things the user explicitly asked MAX to remember
                   ("always keep answers short", "call me Sara").
                   Separate from long_term because preferences change
                   MAX's *behavior*, not just its knowledge.
"""

import sqlite3
import os
import json
from datetime import datetime
from dataclasses import dataclass, field

from backend.config.settings import settings


def _ensure_db():
    os.makedirs(os.path.dirname(settings.MEMORY_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(settings.MEMORY_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS long_term_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            created_at TEXT
        )
    """)
    conn.commit()
    return conn


# NOTE: _ensure_db() is called at the start of every function below (not
# just once at import) so the schema always exists against whatever
# settings.MEMORY_DB_PATH currently points to — matters for tests that
# swap the DB path after this module is first imported.


# ---------------------------------------------------------------------------
# Short-term memory — in-process, per session, conversation window
# ---------------------------------------------------------------------------
@dataclass
class ShortTermMemory:
    max_turns: int = 8
    _sessions: dict = field(default_factory=dict)

    def add_turn(self, session_id: str, role: str, content: str):
        self._sessions.setdefault(session_id, [])
        self._sessions[session_id].append({"role": role, "content": content})
        self._sessions[session_id] = self._sessions[session_id][-self.max_turns:]

    def get_history(self, session_id: str) -> list[dict]:
        return self._sessions.get(session_id, [])

    def clear(self, session_id: str):
        self._sessions.pop(session_id, None)


# Module-level singleton — shared across requests within the running process.
short_term_memory = ShortTermMemory()


# ---------------------------------------------------------------------------
# Long-term memory — persisted, explicit, inspectable, deletable
# ---------------------------------------------------------------------------
def add_long_term_memory(content: str, category: str = "general") -> int:
    conn = _ensure_db()
    cur = conn.execute(
        "INSERT INTO long_term_memory (content, category, created_at) VALUES (?, ?, ?)",
        (content, category, datetime.now().isoformat()),
    )
    conn.commit()
    memory_id = cur.lastrowid
    conn.close()
    return memory_id


def list_long_term_memory() -> list[dict]:
    conn = _ensure_db()
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM long_term_memory ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_long_term_memory(memory_id: int) -> bool:
    conn = _ensure_db()
    cur = conn.execute("DELETE FROM long_term_memory WHERE id = ?", (memory_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def search_long_term_memory(keyword: str) -> list[dict]:
    """Simple keyword search — used by the memory agent to check relevance
    before injecting long-term memory into a prompt."""
    conn = _ensure_db()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM long_term_memory WHERE content LIKE ? ORDER BY id DESC",
        (f"%{keyword}%",),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Preferences — explicit user instructions about MAX's own behavior
# ---------------------------------------------------------------------------
def add_preference(content: str) -> int:
    conn = _ensure_db()
    cur = conn.execute(
        "INSERT INTO preferences (content, created_at) VALUES (?, ?)",
        (content, datetime.now().isoformat()),
    )
    conn.commit()
    pref_id = cur.lastrowid
    conn.close()
    return pref_id


def list_preferences() -> list[dict]:
    conn = _ensure_db()
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM preferences ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_preference(pref_id: int) -> bool:
    conn = _ensure_db()
    cur = conn.execute("DELETE FROM preferences WHERE id = ?", (pref_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted
