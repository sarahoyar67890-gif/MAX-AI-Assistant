"""
API routes. Kept thin — orchestration logic lives in the graph/agents,
this layer just validates input, invokes the graph, and shapes output.
"""

import time
import os
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Optional

from backend.models.schemas import ChatRequest, ChatResponse, PerformanceInfo, ApprovalRequest
from backend.graph.graph import max_graph
from backend.memory.memory_store import (
    short_term_memory, list_long_term_memory, delete_long_term_memory,
    add_preference, list_preferences, delete_preference,
)
from backend.services.observability import get_recent_traces, get_session_trace, get_node_performance_summary, is_langsmith_enabled
from backend.services.llm_service import LLMError
from backend.rag.ingestion import ingest_file, chunk_text
from backend.rag.retrieval import HybridRetriever

router = APIRouter()

_retriever = None
def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever

# In-memory pending-approval store: session_id -> pending action details.
# NOTE: process-local, resets on restart — fine for a single-instance
# portfolio deployment; swap for Redis/DB if this ever runs multi-instance.
_pending_approvals: dict[str, dict] = {}


@router.get("/health")
def health_check():
    return {
        "status": "MAX is online",
        "langsmith_enabled": is_langsmith_enabled(),
        "knowledge_base_documents": get_retriever().document_count(),
    }


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    start = time.perf_counter()
    history = short_term_memory.get_history(request.session_id)

    try:
        result = max_graph.invoke({
            "user_input": request.message,
            "session_id": request.session_id,
            "conversation_history": history,
            "llm_call_count": 0,
            "token_estimate": 0,
            "tools_used": [],
            "sources": [],
        })
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e))

    short_term_memory.add_turn(request.session_id, "user", request.message)
    short_term_memory.add_turn(request.session_id, "assistant", result.get("response", ""))

    if result.get("requires_approval"):
        _pending_approvals[request.session_id] = {"action": result.get("approval_action"), "original_message": request.message}

    latency = time.perf_counter() - start

    return ChatResponse(
        response=result.get("response", ""),
        requires_approval=bool(result.get("requires_approval")),
        approval_action=result.get("approval_action"),
        sources=result.get("sources", []),
        performance=PerformanceInfo(
            latency_seconds=round(latency, 2),
            llm_calls=result.get("llm_call_count", 0),
            tokens_used=result.get("token_estimate", 0),
            route=result.get("intent", "unknown"),
            tools_used=result.get("tools_used", []),
        ),
    )


@router.post("/approve")
def approve_action(request: ApprovalRequest):
    """
    Human-in-the-loop endpoint. If a sensitive tool (see
    settings.SENSITIVE_TOOLS) was requested, the /chat call ends early
    with requires_approval=True instead of executing it. The frontend
    shows the user the exact action and calls this endpoint with their
    decision before anything sensitive actually runs.
    """
    pending = _pending_approvals.pop(request.session_id, None)
    if not pending:
        raise HTTPException(status_code=404, detail="No pending approval found for this session.")

    if not request.approved:
        return {"status": "rejected", "message": "Action was not approved and was not executed."}

    # Real execution of the one currently-wired sensitive tool. Additional
    # sensitive tools would branch here the same way as they're added.
    action = pending["action"]
    if action.startswith("send_email"):
        return {"status": "approved", "message": "Email drafting approved — note send_email only prepares a draft in this build; no live email service is connected."}

    return {"status": "approved", "message": f"Approved: {action}"}


@router.get("/traces/recent")
def recent_traces(limit: int = 50):
    return {"traces": get_recent_traces(limit)}


@router.get("/traces/session/{session_id}")
def session_trace(session_id: str):
    return {"session_id": session_id, "trace": get_session_trace(session_id)}


@router.get("/traces/performance-summary")
def performance_summary():
    return {"node_performance": get_node_performance_summary()}


@router.get("/memory/long-term")
def get_long_term_memory():
    return {"memories": list_long_term_memory()}


@router.delete("/memory/long-term/{memory_id}")
def delete_memory(memory_id: int):
    deleted = delete_long_term_memory(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return {"status": "deleted", "memory_id": memory_id}


@router.get("/memory/preferences")
def get_preferences():
    return {"preferences": list_preferences()}


@router.post("/memory/preferences")
def add_pref(content: str):
    pref_id = add_preference(content)
    return {"status": "added", "preference_id": pref_id}


@router.delete("/memory/preferences/{pref_id}")
def delete_pref(pref_id: int):
    deleted = delete_preference(pref_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Preference not found.")
    return {"status": "deleted", "preference_id": pref_id}


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith((".txt", ".md")):
        raise HTTPException(
            status_code=400,
            detail="Only .txt and .md files are supported in this build. "
                   "PDF/docx ingestion needs an extraction step not wired up yet.",
        )
    os.makedirs("./data/uploads", exist_ok=True)
    save_path = f"./data/uploads/{file.filename}"
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    chunks = ingest_file(save_path)
    get_retriever().add_chunks(chunks)

    return {"status": "indexed", "filename": file.filename, "chunks_created": len(chunks)}


@router.get("/documents")
def list_documents():
    return {"sources": get_retriever().list_sources(), "total_chunks": get_retriever().document_count()}


@router.get("/tasks")
def get_tasks(include_completed: bool = False):
    from backend.tools.productivity_tools import list_tasks
    return {"tasks": list_tasks.invoke({"include_completed": include_completed})}
