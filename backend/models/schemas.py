"""
Shared data models — request/response schemas for the API and
structured-output schemas the LLM is asked to fill in.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal


# --- API request/response ---

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class PerformanceInfo(BaseModel):
    latency_seconds: float
    llm_calls: int
    tokens_used: int
    route: str
    tools_used: list[str] = []


class ChatResponse(BaseModel):
    response: str
    requires_approval: bool = False
    approval_action: Optional[str] = None
    sources: list[str] = []
    performance: PerformanceInfo


class ApprovalRequest(BaseModel):
    session_id: str
    approved: bool


# --- Structured outputs the LLM fills in ---

class IntentClassification(BaseModel):
    """Router's structured decision about how to handle a message."""
    intent: Literal["chat", "task", "knowledge", "communication"] = Field(
        description="chat = casual conversation; task = productivity action "
                    "(reminders, planning, checklist); knowledge = needs "
                    "retrieval from the user's documents; communication = "
                    "drafting/rewriting/summarizing text"
    )
    reasoning: str = Field(description="One short sentence on why this intent was chosen")


class RetrievalConfidence(BaseModel):
    """Self-assessment of whether retrieved context is sufficient to answer."""
    has_sufficient_evidence: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str


class VerificationResult(BaseModel):
    """Verification node's check on whether a response is grounded/safe."""
    is_grounded: bool = Field(description="True if every claim is supported by retrieved evidence or tool output")
    is_safe: bool = Field(description="True if response contains no unsafe/inappropriate content")
    issues: list[str] = Field(default_factory=list, description="Specific problems found, if any")
