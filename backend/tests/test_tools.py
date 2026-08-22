"""
Tool tests. Productivity tools are pure logic/DB — tested for real.
Communication tools call the LLM — mocked, since real calls need a
Groq key not available here.
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


@pytest.fixture
def productivity_tools(tmp_path, monkeypatch):
    from backend.config import settings as settings_module
    monkeypatch.setattr(settings_module.settings, "MEMORY_DB_PATH", str(tmp_path / "memory.db"))
    import backend.tools.productivity_tools as pt
    return pt


class TestProductivityTools:
    def test_create_task(self, productivity_tools):
        result = productivity_tools.create_task.invoke({"title": "Submit assignment", "priority": "high", "deadline": "tonight"})
        assert "Submit assignment" in result
        assert "high" in result

    def test_list_tasks_empty(self, productivity_tools):
        result = productivity_tools.list_tasks.invoke({"include_completed": False})
        assert "No tasks" in result

    def test_list_tasks_shows_created_task(self, productivity_tools):
        productivity_tools.create_task.invoke({"title": "Buy groceries", "priority": "medium", "deadline": ""})
        result = productivity_tools.list_tasks.invoke({"include_completed": False})
        assert "Buy groceries" in result

    def test_prioritize_tasks_orders_high_first(self, productivity_tools):
        productivity_tools.create_task.invoke({"title": "Low priority thing", "priority": "low", "deadline": ""})
        productivity_tools.create_task.invoke({"title": "Urgent thing", "priority": "high", "deadline": ""})
        result = productivity_tools.prioritize_tasks.invoke({})
        # "Urgent thing" (high) should appear before "Low priority thing" (low)
        assert result.index("Urgent thing") < result.index("Low priority thing")

    def test_prioritize_no_tasks(self, productivity_tools):
        result = productivity_tools.prioritize_tasks.invoke({})
        assert "No open tasks" in result

    def test_generate_checklist(self, productivity_tools):
        result = productivity_tools.generate_checklist.invoke({
            "goal": "Deploy the app",
            "steps": ["Write tests", "Build Docker image", "Push to registry"],
        })
        assert "Deploy the app" in result
        assert "Write tests" in result
        assert result.count("[ ]") == 3

    def test_tasks_persist_across_calls(self, productivity_tools):
        productivity_tools.create_task.invoke({"title": "Task A", "priority": "medium", "deadline": ""})
        productivity_tools.create_task.invoke({"title": "Task B", "priority": "medium", "deadline": ""})
        result = productivity_tools.list_tasks.invoke({"include_completed": False})
        assert "Task A" in result and "Task B" in result


class TestCommunicationTools:
    def test_draft_message_calls_llm(self):
        from backend.tools.communication_tools import draft_message
        fake_response = MagicMock()
        fake_response.content = "Hi team, following up on the report."
        with patch("backend.tools.communication_tools.chat_llm", return_value=MagicMock()), \
             patch("backend.tools.communication_tools.invoke_with_retry", return_value=fake_response):
            result = draft_message.invoke({"context": "follow up on the quarterly report", "tone": "professional"})
        assert result == "Hi team, following up on the report."

    def test_rewrite_professionally_calls_llm(self):
        from backend.tools.communication_tools import rewrite_professionally
        fake_response = MagicMock()
        fake_response.content = "I would appreciate your prompt response."
        with patch("backend.tools.communication_tools.chat_llm", return_value=MagicMock()), \
             patch("backend.tools.communication_tools.invoke_with_retry", return_value=fake_response):
            result = rewrite_professionally.invoke({"text": "hey answer me quick"})
        assert result == "I would appreciate your prompt response."

    def test_summarize_text_calls_llm(self):
        from backend.tools.communication_tools import summarize_text
        fake_response = MagicMock()
        fake_response.content = "Short summary."
        with patch("backend.tools.communication_tools.chat_llm", return_value=MagicMock()), \
             patch("backend.tools.communication_tools.invoke_with_retry", return_value=fake_response):
            result = summarize_text.invoke({"text": "A very long article " * 50, "max_sentences": 2})
        assert result == "Short summary."

    def test_send_email_never_claims_to_send(self):
        from backend.tools.communication_tools import send_email
        result = send_email.invoke({"to": "boss@company.com", "subject": "Update", "body": "Here's my update."})
        assert "not sent" in result.lower() or "draft only" in result.lower()
        assert "boss@company.com" in result

    def test_send_email_is_marked_sensitive(self):
        from backend.config.settings import settings
        assert "send_email" in settings.SENSITIVE_TOOLS
