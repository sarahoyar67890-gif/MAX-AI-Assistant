"""
Evaluation dataset. Small, representative, hand-written — not huge, but
real: each case has a defined expected behavior so evaluate.py can score
against something concrete instead of just eyeballing outputs.
"""

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    id: str
    input: str
    expected_intent: str
    category: str  # "chat" | "task" | "knowledge" | "communication"
    expected_tool: str = None  # for task/communication cases with a clear right tool
    expected_keywords: list[str] = field(default_factory=list)  # for knowledge cases — must appear in a faithful answer
    requires_approval_expected: bool = False


EVAL_DATASET: list[EvalCase] = [
    # --- Chat ---
    EvalCase(id="chat_01", input="Hey MAX, how's it going?", expected_intent="chat", category="chat"),
    EvalCase(id="chat_02", input="What do you think makes a good morning routine?", expected_intent="chat", category="chat"),
    EvalCase(id="chat_03", input="Tell me something interesting.", expected_intent="chat", category="chat"),

    # --- Task ---
    EvalCase(id="task_01", input="Remind me to submit my assignment tonight", expected_intent="task",
              category="task", expected_tool="create_task"),
    EvalCase(id="task_02", input="What tasks do I have open right now?", expected_intent="task",
              category="task", expected_tool="list_tasks"),
    EvalCase(id="task_03", input="What should I work on first today?", expected_intent="task",
              category="task", expected_tool="prioritize_tasks"),
    EvalCase(id="task_04", input="What time is it right now?", expected_intent="task",
              category="task", expected_tool="get_current_time"),
    EvalCase(id="task_05", input="What's 340 divided by 4, times 7?", expected_intent="task",
              category="task", expected_tool="calculate"),

    # --- Knowledge (RAG) — depends on documents being ingested first ---
    EvalCase(id="knowledge_01", input="How much is the home office stipend?", expected_intent="knowledge",
              category="knowledge", expected_keywords=["$200", "200"]),
    EvalCase(id="knowledge_02", input="How many remote days per week are allowed?", expected_intent="knowledge",
              category="knowledge", expected_keywords=["3", "three"]),
    EvalCase(id="knowledge_03", input="What's the capital of Australia?", expected_intent="knowledge",
              category="knowledge", expected_keywords=[]),  # should trigger "not enough evidence" — not in the KB

    # --- Communication ---
    EvalCase(id="comm_01", input="Rewrite this professionally: 'hey can u send that over asap'",
              expected_intent="communication", category="communication", expected_tool="rewrite_professionally"),
    EvalCase(id="comm_02", input="Summarize this in one sentence: The quarterly report shows revenue grew 12% "
              "year over year, driven mainly by the enterprise segment, while churn in the SMB segment increased slightly.",
              expected_intent="communication", category="communication", expected_tool="summarize_text"),
    EvalCase(id="comm_03", input="Draft an email to my manager asking for Friday off",
              expected_intent="communication", category="communication", expected_tool="send_email",
              requires_approval_expected=True),
]
