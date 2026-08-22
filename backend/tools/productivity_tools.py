"""
Productivity tools. Task/reminder data persists to SQLite (survives
server restarts, unlike the earlier prototype's in-memory list).
"""

from langchain_core.tools import tool
import sqlite3
import os
from datetime import datetime

from backend.config.settings import settings


def _ensure_db():
    os.makedirs(os.path.dirname(settings.MEMORY_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(settings.MEMORY_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            priority TEXT DEFAULT 'medium',
            deadline TEXT,
            done INTEGER DEFAULT 0,
            created_at TEXT
        )
    """)
    conn.commit()
    return conn


# NOTE: _ensure_db() is called at the start of every function below (not
# just once at import) so the schema is guaranteed to exist against
# whatever settings.MEMORY_DB_PATH currently points to — this matters for
# tests that swap the DB path via monkeypatch after this module is first
# imported, and also makes the module self-healing if the DB file is
# ever deleted while the server is running.


@tool
def create_task(title: str, priority: str = "medium", deadline: str = "") -> str:
    """Create a task/reminder for the user.

    Args:
        title: What the task is.
        priority: 'low', 'medium', or 'high'.
        deadline: When it's due, in the user's own words (optional).
    """
    conn = _ensure_db()
    conn.execute(
        "INSERT INTO tasks (title, priority, deadline, created_at) VALUES (?, ?, ?, ?)",
        (title, priority, deadline, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    return f"Task created: '{title}' (priority: {priority}" + (f", due: {deadline})" if deadline else ")")


@tool
def list_tasks(include_completed: bool = False) -> str:
    """List the user's current tasks.

    Args:
        include_completed: Whether to include already-completed tasks.
    """
    conn = _ensure_db()
    conn.row_factory = sqlite3.Row
    query = "SELECT * FROM tasks" if include_completed else "SELECT * FROM tasks WHERE done = 0"
    rows = conn.execute(query + " ORDER BY priority DESC, id DESC").fetchall()
    conn.close()
    if not rows:
        return "No tasks found."
    lines = []
    for r in rows:
        status = "[done]" if r["done"] else "[ ]"
        deadline = f" (due: {r['deadline']})" if r["deadline"] else ""
        lines.append(f"{status} #{r['id']} [{r['priority']}] {r['title']}{deadline}")
    return "\n".join(lines)


@tool
def prioritize_tasks() -> str:
    """Return the user's open tasks sorted by priority (high -> low), to
    help decide what to work on first."""
    conn = _ensure_db()
    conn.row_factory = sqlite3.Row
    order = "CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END"
    rows = conn.execute(f"SELECT * FROM tasks WHERE done = 0 ORDER BY {order}").fetchall()
    conn.close()
    if not rows:
        return "No open tasks to prioritize."
    lines = [f"{i+1}. [{r['priority']}] {r['title']}" for i, r in enumerate(rows)]
    return "Suggested priority order:\n" + "\n".join(lines)


@tool
def generate_checklist(goal: str, steps: list[str]) -> str:
    """Create a structured checklist for a goal, given a list of steps.

    Args:
        goal: What the checklist is for.
        steps: The individual steps/items.
    """
    lines = [f"Checklist: {goal}"] + [f"  [ ] {s}" for s in steps]
    return "\n".join(lines)


PRODUCTIVITY_TOOLS = [create_task, list_tasks, prioritize_tasks, generate_checklist]
