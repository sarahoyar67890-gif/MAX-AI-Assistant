"""
Shared state that flows through every node in MAX's graph.
"""

from typing import TypedDict, Optional


class MaxState(TypedDict, total=False):
    # input
    user_input: str
    session_id: str
    conversation_history: list[dict]

    # routing
    intent: str
    intent_reasoning: str

    # RAG
    retrieved_chunks: list[dict]
    retrieval_confidence: float
    sources: list[str]

    # tool execution
    tools_used: list[str]

    # memory
    relevant_memories: list[str]
    memory_to_persist: Optional[str]

    # verification / safety
    is_grounded: bool
    is_safe: bool
    verification_issues: list[str]

    # human approval
    requires_approval: bool
    approval_action: Optional[str]
    approved: Optional[bool]

    # output
    response: str

    # performance tracking
    llm_call_count: int
    token_estimate: int
