"""
Agent node implementations.

Each function takes MaxState in, returns a partial state update — the
standard LangGraph node pattern. Every node is wrapped in `trace_node`
so latency/metadata is recorded automatically for observability.
"""

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import json

from backend.graph.state import MaxState
from backend.services.llm_service import fast_llm, chat_llm, invoke_with_retry, estimate_tokens, LLMError
from backend.services.observability import trace_node
from backend.config.settings import settings
from backend.models.schemas import IntentClassification, VerificationResult

from backend.tools.productivity_tools import PRODUCTIVITY_TOOLS
from backend.tools.communication_tools import COMMUNICATION_TOOLS
from backend.rag.retrieval import HybridRetriever, retrieval_confidence
from backend.memory.memory_store import (
    short_term_memory, search_long_term_memory, add_long_term_memory, list_preferences,
)

ALL_TOOLS = PRODUCTIVITY_TOOLS + COMMUNICATION_TOOLS
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

# Retriever is expensive to construct (loads models) — build once, reuse.
_retriever_singleton = None
def get_retriever() -> HybridRetriever:
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = HybridRetriever()
    return _retriever_singleton


# ---------------------------------------------------------------------------
# 1. Intent analysis / router node
# ---------------------------------------------------------------------------
def router_node(state: MaxState) -> dict:
    with trace_node("router_node", session_id=state.get("session_id", "default")):
        llm = fast_llm(temperature=0.0).with_structured_output(IntentClassification)
        system = SystemMessage(content=(
            "Classify the user's message intent. Consider the conversation "
            "history for context (e.g. follow-up questions)."
        ))
        history_text = "\n".join(
            f"{h['role']}: {h['content']}" for h in state.get("conversation_history", [])[-4:]
        )
        human = HumanMessage(content=f"Recent history:\n{history_text}\n\nMessage: {state['user_input']}")

        try:
            result: IntentClassification = llm.invoke([system, human])
            intent, reasoning = result.intent, result.reasoning
        except Exception as e:
            # Fallback: default to chat rather than crash the whole graph
            intent, reasoning = "chat", f"Router failed ({e}), defaulting to chat."

        return {
            "intent": intent,
            "intent_reasoning": reasoning,
            "llm_call_count": state.get("llm_call_count", 0) + 1,
        }


def route_decision(state: MaxState) -> str:
    return {
        "chat": "chat_agent",
        "task": "task_agent",
        "knowledge": "rag_agent",
        "communication": "task_agent",  # communication tools live in the same tool-calling node as task tools
    }.get(state["intent"], "chat_agent")


# ---------------------------------------------------------------------------
# 2. Memory agent — checks for relevant long-term memory + preferences
#    (runs before the specialist agents so they can use it as context)
# ---------------------------------------------------------------------------
def memory_agent_node(state: MaxState) -> dict:
    with trace_node("memory_agent_node", session_id=state.get("session_id", "default")):
        relevant = search_long_term_memory(state["user_input"][:50])  # simple keyword pass
        prefs = list_preferences()
        memory_strs = [m["content"] for m in relevant[:3]]
        pref_strs = [p["content"] for p in prefs]
        return {"relevant_memories": memory_strs + pref_strs}


# ---------------------------------------------------------------------------
# 3. Chat agent
# ---------------------------------------------------------------------------
def chat_agent_node(state: MaxState) -> dict:
    with trace_node("chat_agent_node", session_id=state.get("session_id", "default")):
        llm = chat_llm(temperature=0.6)
        memory_context = ""
        if state.get("relevant_memories"):
            memory_context = "\nRelevant things you know about the user:\n" + "\n".join(
                f"- {m}" for m in state["relevant_memories"]
            )
        system = SystemMessage(content=(
            "You are MAX, a warm, helpful, capable personal AI assistant. "
            "Keep responses natural and concise (2-4 sentences) unless detail is requested."
            f"{memory_context}"
        ))
        human = HumanMessage(content=state["user_input"])
        try:
            result = invoke_with_retry(llm, [system, human])
            response = result.content
        except LLMError as e:
            response = f"I'm having trouble reaching my language model right now ({e}). Please try again shortly."

        return {
            "response": response,
            "llm_call_count": state.get("llm_call_count", 0) + 1,
            "token_estimate": state.get("token_estimate", 0) + estimate_tokens(response),
        }


# ---------------------------------------------------------------------------
# 4. Task agent — real tool calling (productivity + communication tools)
# ---------------------------------------------------------------------------
def task_agent_node(state: MaxState) -> dict:
    with trace_node("task_agent_node", session_id=state.get("session_id", "default")):
        llm = chat_llm(temperature=0.2).bind_tools(ALL_TOOLS)
        system = SystemMessage(content=(
            "You are MAX, a helpful assistant with access to productivity and "
            "communication tools. Use a tool when the request needs one. "
            "If you cannot actually perform an action (e.g. no tool exists for it), "
            "say so honestly instead of pretending you did it. "
            "After a tool result, reply naturally confirming what happened."
        ))
        human = HumanMessage(content=state["user_input"])
        llm_calls = state.get("llm_call_count", 0)

        try:
            ai_message = invoke_with_retry(llm, [system, human])
        except LLMError as e:
            return {"response": f"I couldn't process that task right now ({e}).", "llm_call_count": llm_calls + 1}

        llm_calls += 1
        tools_used = []

        if not ai_message.tool_calls:
            return {
                "response": ai_message.content,
                "llm_call_count": llm_calls,
                "tools_used": tools_used,
            }

        # Check if any requested tool is sensitive -> require human approval
        # BEFORE executing it, rather than after.
        sensitive_calls = [c for c in ai_message.tool_calls if c["name"] in settings.SENSITIVE_TOOLS]
        if sensitive_calls:
            action_desc = ", ".join(f"{c['name']}({c['args']})" for c in sensitive_calls)
            return {
                "requires_approval": True,
                "approval_action": action_desc,
                "response": f"This requires your approval before I proceed: {action_desc}",
                "llm_call_count": llm_calls,
            }

        tool_messages = []
        for call in ai_message.tool_calls:
            tool_fn = TOOLS_BY_NAME.get(call["name"])
            result = tool_fn.invoke(call["args"]) if tool_fn else f"Unknown tool: {call['name']}"
            tools_used.append(call["name"])
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

        try:
            final = invoke_with_retry(llm, [system, human, ai_message, *tool_messages])
            response = final.content
        except LLMError as e:
            response = f"I ran the tool(s) but couldn't compose a final reply ({e})."
        llm_calls += 1

        return {
            "response": response,
            "llm_call_count": llm_calls,
            "tools_used": tools_used,
            "token_estimate": state.get("token_estimate", 0) + estimate_tokens(response),
        }


# ---------------------------------------------------------------------------
# 5. RAG agent — advanced retrieval with confidence-gated answering
# ---------------------------------------------------------------------------
def rag_agent_node(state: MaxState) -> dict:
    with trace_node("rag_agent_node", session_id=state.get("session_id", "default")):
        retriever = get_retriever()
        llm_calls = state.get("llm_call_count", 0)

        if retriever.document_count() == 0:
            return {
                "response": "I don't have any documents in my knowledge base yet — upload some first and I'll be able to answer from them.",
                "retrieval_confidence": 0.0,
                "sources": [],
                "llm_call_count": llm_calls,
            }

        results = retriever.retrieve(state["user_input"])
        confidence = retrieval_confidence(results)

        if confidence < settings.MIN_RETRIEVAL_CONFIDENCE or not results:
            return {
                "response": "I don't have enough evidence in my knowledge base to answer that confidently — I'd rather tell you that than guess.",
                "retrieval_confidence": confidence,
                "sources": [],
                "llm_call_count": llm_calls,
            }

        context = "\n\n".join(f"[Source: {r.source}]\n{r.text}" for r in results)
        llm = chat_llm(temperature=0.2)
        system = SystemMessage(content=(
            "Answer the user's question using ONLY the provided context. "
            "If the context doesn't fully answer it, say what's missing. "
            "Cite which source(s) you used."
        ))
        human = HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['user_input']}")

        try:
            result = invoke_with_retry(llm, [system, human])
            response = result.content
        except LLMError as e:
            response = f"I retrieved relevant context but couldn't generate an answer ({e})."
        llm_calls += 1

        return {
            "response": response,
            "retrieval_confidence": confidence,
            "sources": list({r.source for r in results}),
            "retrieved_chunks": [r.to_dict() for r in results],
            "llm_call_count": llm_calls,
            "token_estimate": state.get("token_estimate", 0) + estimate_tokens(context) + estimate_tokens(response),
        }


# ---------------------------------------------------------------------------
# 6. Verification node — checks groundedness + safety before returning
# ---------------------------------------------------------------------------
def verification_node(state: MaxState) -> dict:
    with trace_node("verification_node", session_id=state.get("session_id", "default")):
        # Cheap heuristic pass first: skip LLM verification entirely for
        # short/simple chat responses where grounding isn't meaningful —
        # this is a real latency/cost optimization, not just a fallback.
        if state["intent"] == "chat" and len(state.get("response", "")) < 300:
            return {"is_grounded": True, "is_safe": True, "verification_issues": []}

        llm = fast_llm(temperature=0.0).with_structured_output(VerificationResult)
        context_note = ""
        if state.get("retrieved_chunks"):
            context_note = "Response should be grounded in the retrieved context provided earlier in this session."

        system = SystemMessage(content=(
            "Check this AI assistant response for: (1) groundedness — are claims "
            "supported by context/tool output rather than invented, and (2) safety — "
            "no harmful, inappropriate, or misleading content. " + context_note
        ))
        human = HumanMessage(content=f"Response to check:\n{state.get('response', '')}")

        try:
            result: VerificationResult = llm.invoke([system, human])
            return {
                "is_grounded": result.is_grounded,
                "is_safe": result.is_safe,
                "verification_issues": result.issues,
                "llm_call_count": state.get("llm_call_count", 0) + 1,
            }
        except Exception:
            # Fail open on verification errors — don't block a response over
            # a verification-model hiccup — but log it as an issue for observability.
            return {"is_grounded": True, "is_safe": True, "verification_issues": ["verification_check_failed"]}


def verification_decision(state: MaxState) -> str:
    if not state.get("is_safe", True):
        return "guardrail_block"
    return "finalize"


# ---------------------------------------------------------------------------
# 7. Guardrail block node — replaces response if verification failed safety
# ---------------------------------------------------------------------------
def guardrail_block_node(state: MaxState) -> dict:
    with trace_node("guardrail_block_node", session_id=state.get("session_id", "default")):
        issues = ", ".join(state.get("verification_issues", [])) or "a safety concern"
        return {
            "response": f"I'm not comfortable sharing that response as generated — it flagged {issues}. Could you rephrase what you need?"
        }


# ---------------------------------------------------------------------------
# 8. Memory persistence — decides whether to save anything from this turn
# ---------------------------------------------------------------------------
MEMORY_WORTHY_KEYWORDS = ("remember", "my name is", "i prefer", "i like", "i work at", "i live in")

def maybe_persist_memory_node(state: MaxState) -> dict:
    with trace_node("memory_persist_node", session_id=state.get("session_id", "default")):
        text_lower = state["user_input"].lower()
        if any(kw in text_lower for kw in MEMORY_WORTHY_KEYWORDS):
            add_long_term_memory(state["user_input"], category="user_stated")
            return {"memory_to_persist": state["user_input"]}
        return {}
