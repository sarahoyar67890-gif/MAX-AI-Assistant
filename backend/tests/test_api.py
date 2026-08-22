"""
API-layer tests. The LangGraph invocation itself is mocked here — it
needs a real GROQ_API_KEY to run for real, which isn't available in this
test environment. What's tested for real: request validation, response
shaping, the approval workflow state machine, and every memory/document
CRUD endpoint (those touch real SQLite/Chroma, not mocks).
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


@pytest.fixture
def client(tmp_path, monkeypatch):
    from backend.config import settings as settings_module
    monkeypatch.setattr(settings_module.settings, "MEMORY_DB_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setattr(settings_module.settings, "TRACE_DB_PATH", str(tmp_path / "traces.db"))
    monkeypatch.setattr(settings_module.settings, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))

    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)


class TestHealthAndRoot:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "MAX is online"

    def test_health_check(self, client):
        with patch("backend.api.routes.get_retriever") as mock_get_retriever:
            mock_retriever = MagicMock()
            mock_retriever.document_count.return_value = 0
            mock_get_retriever.return_value = mock_retriever
            r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert "knowledge_base_documents" in body
        assert "langsmith_enabled" in body


class TestChatEndpoint:
    def test_empty_message_rejected(self, client):
        r = client.post("/chat", json={"message": "", "session_id": "s1"})
        assert r.status_code == 400

    def test_valid_message_returns_shaped_response(self, client):
        fake_result = {
            "response": "Hi there!",
            "intent": "chat",
            "llm_call_count": 1,
            "token_estimate": 20,
            "tools_used": [],
            "sources": [],
        }
        with patch("backend.api.routes.max_graph") as mock_graph:
            mock_graph.invoke.return_value = fake_result
            r = client.post("/chat", json={"message": "hello", "session_id": "s1"})

        assert r.status_code == 200
        body = r.json()
        assert body["response"] == "Hi there!"
        assert body["performance"]["route"] == "chat"
        assert body["performance"]["llm_calls"] == 1
        assert body["requires_approval"] is False

    def test_sensitive_action_flags_approval(self, client):
        fake_result = {
            "response": "This requires your approval: send_email(...)",
            "intent": "communication",
            "requires_approval": True,
            "approval_action": "send_email({'to': 'a@b.com'})",
            "llm_call_count": 1,
            "tools_used": [],
            "sources": [],
        }
        with patch("backend.api.routes.max_graph") as mock_graph:
            mock_graph.invoke.return_value = fake_result
            r = client.post("/chat", json={"message": "email my boss", "session_id": "s2"})

        assert r.status_code == 200
        body = r.json()
        assert body["requires_approval"] is True
        assert "send_email" in body["approval_action"]


class TestApprovalWorkflow:
    def test_approve_without_pending_returns_404(self, client):
        r = client.post("/approve", json={"session_id": "nonexistent", "approved": True})
        assert r.status_code == 404

    def test_full_approval_flow(self, client):
        fake_result = {
            "response": "needs approval",
            "intent": "communication",
            "requires_approval": True,
            "approval_action": "send_email({'to': 'boss@co.com'})",
            "llm_call_count": 1,
            "tools_used": [],
            "sources": [],
        }
        with patch("backend.api.routes.max_graph") as mock_graph:
            mock_graph.invoke.return_value = fake_result
            client.post("/chat", json={"message": "email my boss", "session_id": "s3"})

        r = client.post("/approve", json={"session_id": "s3", "approved": True})
        assert r.status_code == 200
        assert r.json()["status"] == "approved"

    def test_rejection_does_not_execute(self, client):
        fake_result = {
            "response": "needs approval",
            "intent": "communication",
            "requires_approval": True,
            "approval_action": "send_email({'to': 'boss@co.com'})",
            "llm_call_count": 1,
            "tools_used": [],
            "sources": [],
        }
        with patch("backend.api.routes.max_graph") as mock_graph:
            mock_graph.invoke.return_value = fake_result
            client.post("/chat", json={"message": "email my boss", "session_id": "s4"})

        r = client.post("/approve", json={"session_id": "s4", "approved": False})
        assert r.status_code == 200
        assert r.json()["status"] == "rejected"


class TestMemoryEndpoints:
    def test_add_and_list_long_term_memory(self, client):
        from backend.memory.memory_store import add_long_term_memory
        add_long_term_memory("User's favorite color is blue.")
        r = client.get("/memory/long-term")
        assert r.status_code == 200
        assert len(r.json()["memories"]) == 1

    def test_delete_long_term_memory(self, client):
        from backend.memory.memory_store import add_long_term_memory
        mem_id = add_long_term_memory("Temporary fact.")
        r = client.delete(f"/memory/long-term/{mem_id}")
        assert r.status_code == 200
        r2 = client.get("/memory/long-term")
        assert len(r2.json()["memories"]) == 0

    def test_delete_nonexistent_memory_404(self, client):
        r = client.delete("/memory/long-term/99999")
        assert r.status_code == 404

    def test_preferences_crud(self, client):
        r = client.post("/memory/preferences", params={"content": "Keep answers short."})
        assert r.status_code == 200
        pref_id = r.json()["preference_id"]

        r2 = client.get("/memory/preferences")
        assert len(r2.json()["preferences"]) == 1

        r3 = client.delete(f"/memory/preferences/{pref_id}")
        assert r3.status_code == 200


class TestDocumentEndpoints:
    def test_upload_rejects_unsupported_type(self, client):
        r = client.post(
            "/documents/upload",
            files={"file": ("test.pdf", b"fake pdf bytes", "application/pdf")},
        )
        assert r.status_code == 400

    def test_upload_and_list_txt_document(self, client, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        from chromadb.api.types import EmbeddingFunction

        class _FakeEF(EmbeddingFunction):
            def __init__(self): pass
            def __call__(self, input):
                return [[float(hash(t) % 997) / 997.0] * 8 for t in input]
            def name(self): return "default"
            @staticmethod
            def build_from_config(config): return _FakeEF()
            def get_config(self): return {}

        with patch("backend.rag.retrieval.embedding_functions.SentenceTransformerEmbeddingFunction") as mock_cls:
            mock_cls.return_value = _FakeEF()
            with patch("backend.api.routes._retriever", None):
                r = client.post(
                    "/documents/upload",
                    files={"file": ("policy.txt", b"Remote work is allowed three days a week.", "text/plain")},
                )
        assert r.status_code == 200
        assert r.json()["chunks_created"] >= 1


class TestTasksEndpoint:
    def test_list_tasks_empty(self, client):
        r = client.get("/tasks")
        assert r.status_code == 200
        assert "No tasks" in r.json()["tasks"]
