"""
Full graph assembly.

    User -> Router -> Memory Agent -> [Chat | Task | RAG] Agent
         -> Verification -> (Guardrail Block | Finalize)
         -> Memory Persist -> END

Human approval is handled OUTSIDE the graph (in the API layer): if
task_agent_node sets requires_approval=True, the graph ends there and
the API returns that to the frontend. A second API call (after the user
approves/rejects) re-enters the graph via a dedicated execute-approved-
action path. This keeps the graph itself simple and avoids blocking
graph execution on a human being present synchronously.
"""

from langgraph.graph import StateGraph, END

from backend.graph.state import MaxState
from backend.agents.nodes import (
    router_node, route_decision,
    memory_agent_node,
    chat_agent_node, task_agent_node, rag_agent_node,
    verification_node, verification_decision,
    guardrail_block_node,
    maybe_persist_memory_node,
)


def build_max_graph():
    graph = StateGraph(MaxState)

    graph.add_node("router", router_node)
    graph.add_node("memory_agent", memory_agent_node)
    graph.add_node("chat_agent", chat_agent_node)
    graph.add_node("task_agent", task_agent_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("verification", verification_node)
    graph.add_node("guardrail_block", guardrail_block_node)
    graph.add_node("memory_persist", maybe_persist_memory_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "memory_agent")

    graph.add_conditional_edges(
        "memory_agent",
        route_decision,
        {"chat_agent": "chat_agent", "task_agent": "task_agent", "rag_agent": "rag_agent"},
    )

    # All specialist agents converge on verification, UNLESS the task agent
    # flagged a sensitive action needing approval — that short-circuits
    # straight to END so the API can surface the approval request.
    def post_agent_decision(state: MaxState) -> str:
        if state.get("requires_approval"):
            return "end_for_approval"
        return "verification"

    graph.add_conditional_edges("chat_agent", post_agent_decision, {"verification": "verification", "end_for_approval": END})
    graph.add_conditional_edges("task_agent", post_agent_decision, {"verification": "verification", "end_for_approval": END})
    graph.add_conditional_edges("rag_agent", post_agent_decision, {"verification": "verification", "end_for_approval": END})

    graph.add_conditional_edges(
        "verification",
        verification_decision,
        {"guardrail_block": "guardrail_block", "finalize": "memory_persist"},
    )
    graph.add_edge("guardrail_block", "memory_persist")
    graph.add_edge("memory_persist", END)

    return graph.compile()


max_graph = build_max_graph()
