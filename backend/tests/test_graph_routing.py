"""
Graph/routing tests. LLM calls are mocked (structured-output classification,
chat responses) — what's tested for real is the routing DECISION LOGIC,
the sensitive-action approval short-circuit, and the verification->guardrail
branching, all of which are pure Python and don't need Groq.
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.agents.nodes import route_decision, verification_decision


class TestRouteDecision:
    def test_chat_intent_routes_to_chat_agent(self):
        assert route_decision({"intent": "chat"}) == "chat_agent"

    def test_task_intent_routes_to_task_agent(self):
        assert route_decision({"intent": "task"}) == "task_agent"

    def test_knowledge_intent_routes_to_rag_agent(self):
        assert route_decision({"intent": "knowledge"}) == "rag_agent"

    def test_communication_intent_routes_to_task_agent(self):
        # Communication tools live in the same tool-calling node as task tools
        assert route_decision({"intent": "communication"}) == "task_agent"

    def test_unknown_intent_defaults_to_chat(self):
        assert route_decision({"intent": "something_unexpected"}) == "chat_agent"


class TestVerificationDecision:
    def test_unsafe_response_routes_to_guardrail(self):
        assert verification_decision({"is_safe": False}) == "guardrail_block"

    def test_safe_response_routes_to_finalize(self):
        assert verification_decision({"is_safe": True}) == "finalize"

    def test_missing_is_safe_defaults_safe(self):
        # fail-open default, matches verification_node's own fail-open behavior
        assert verification_decision({}) == "finalize"


class TestRouterNode:
    def test_router_classifies_and_counts_llm_call(self):
        from backend.agents.nodes import router_node
        from backend.models.schemas import IntentClassification

        fake_classification = IntentClassification(intent="task", reasoning="User wants to create a reminder.")
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = fake_classification

        with patch("backend.agents.nodes.fast_llm") as mock_fast_llm:
            mock_fast_llm.return_value.with_structured_output.return_value = fake_llm
            result = router_node({"user_input": "remind me to call mom", "session_id": "s1", "conversation_history": [], "llm_call_count": 0})

        assert result["intent"] == "task"
        assert result["llm_call_count"] == 1

    def test_router_fails_open_to_chat_on_error(self):
        from backend.agents.nodes import router_node

        with patch("backend.agents.nodes.fast_llm") as mock_fast_llm:
            mock_fast_llm.return_value.with_structured_output.return_value.invoke.side_effect = Exception("API down")
            result = router_node({"user_input": "hello", "session_id": "s1", "conversation_history": [], "llm_call_count": 0})

        # Must not crash the graph — falls back to chat
        assert result["intent"] == "chat"


class TestTaskAgentSensitiveActionGating:
    def test_sensitive_tool_call_requires_approval_before_execution(self):
        from backend.agents.nodes import task_agent_node

        fake_tool_call = {"name": "send_email", "args": {"to": "boss@co.com"}, "id": "call_1"}
        fake_ai_message = MagicMock()
        fake_ai_message.tool_calls = [fake_tool_call]

        with patch("backend.agents.nodes.chat_llm") as mock_chat_llm, \
             patch("backend.agents.nodes.invoke_with_retry", return_value=fake_ai_message), \
             patch("backend.agents.nodes.TOOLS_BY_NAME", {"send_email": MagicMock()}) as mock_tools:

            mock_chat_llm.return_value.bind_tools.return_value = MagicMock()
            result = task_agent_node({"user_input": "email my boss about the deadline", "session_id": "s1", "llm_call_count": 0})

        assert result["requires_approval"] is True
        assert "send_email" in result["approval_action"]
        # The sensitive tool itself must NOT have been invoked
        mock_tools["send_email"].invoke.assert_not_called()

    def test_non_sensitive_tool_executes_directly(self):
        from backend.agents.nodes import task_agent_node

        fake_tool_call = {"name": "get_current_time", "args": {}, "id": "call_1"}
        fake_ai_message = MagicMock()
        fake_ai_message.tool_calls = [fake_tool_call]
        fake_final_message = MagicMock()
        fake_final_message.content = "It's currently 3 PM."

        fake_tool = MagicMock()
        fake_tool.invoke.return_value = "3:00 PM"

        with patch("backend.agents.nodes.chat_llm") as mock_chat_llm, \
             patch("backend.agents.nodes.invoke_with_retry", side_effect=[fake_ai_message, fake_final_message]), \
             patch("backend.agents.nodes.TOOLS_BY_NAME", {"get_current_time": fake_tool}):

            mock_chat_llm.return_value.bind_tools.return_value = MagicMock()
            result = task_agent_node({"user_input": "what time is it?", "session_id": "s1", "llm_call_count": 0})

        assert result.get("requires_approval") is None or result.get("requires_approval") is False
        assert "get_current_time" in result["tools_used"]
        fake_tool.invoke.assert_called_once()


class TestRagAgentConfidenceGating:
    def test_empty_knowledge_base_responds_honestly(self):
        from backend.agents.nodes import rag_agent_node

        fake_retriever = MagicMock()
        fake_retriever.document_count.return_value = 0

        with patch("backend.agents.nodes.get_retriever", return_value=fake_retriever):
            result = rag_agent_node({"user_input": "what's our leave policy?", "session_id": "s1", "llm_call_count": 0})

        assert "don't have any documents" in result["response"]
        assert result["retrieval_confidence"] == 0.0

    def test_low_confidence_refuses_to_answer_instead_of_guessing(self):
        from backend.agents.nodes import rag_agent_node

        fake_retriever = MagicMock()
        fake_retriever.document_count.return_value = 5
        fake_retriever.retrieve.return_value = [MagicMock(source="doc.txt", text="irrelevant")]

        with patch("backend.agents.nodes.get_retriever", return_value=fake_retriever), \
             patch("backend.agents.nodes.retrieval_confidence", return_value=0.1):  # below MIN_RETRIEVAL_CONFIDENCE
            result = rag_agent_node({"user_input": "some obscure question", "session_id": "s1", "llm_call_count": 0})

        assert "don't have enough evidence" in result["response"]
        assert result["sources"] == []
