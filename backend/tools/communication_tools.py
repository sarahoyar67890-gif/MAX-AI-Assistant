"""
Communication tools. These call the LLM themselves (unlike productivity
tools, which are pure logic/DB) because drafting/rewriting genuinely
needs language generation.

`send_email` is intentionally a STUB that never actually sends anything —
wiring real email sending needs SMTP/OAuth credentials that don't exist
in this build. It is listed in SENSITIVE_TOOLS in config so the graph
routes it through human approval regardless, and it honestly reports
that it only prepared a draft rather than claiming to have sent it.
"""

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from backend.services.llm_service import chat_llm, invoke_with_retry


@tool
def draft_message(context: str, tone: str = "professional") -> str:
    """Draft an email or message based on the context/intent described.

    Args:
        context: What the message needs to say / accomplish.
        tone: 'professional', 'casual', 'formal', or 'friendly'.
    """
    llm = chat_llm(temperature=0.5)
    system = SystemMessage(content=(
        f"Draft a {tone}-toned message based on the user's context. "
        "Output only the message text, no preamble or explanation."
    ))
    result = invoke_with_retry(llm, [system, HumanMessage(content=context)])
    return result.content


@tool
def rewrite_professionally(text: str) -> str:
    """Rewrite a piece of text in clear, professional language.

    Args:
        text: The original text to rewrite.
    """
    llm = chat_llm(temperature=0.3)
    system = SystemMessage(content=(
        "Rewrite the given text in clear, professional language. "
        "Preserve the original meaning and intent exactly. "
        "Output only the rewritten text."
    ))
    result = invoke_with_retry(llm, [system, HumanMessage(content=text)])
    return result.content


@tool
def summarize_text(text: str, max_sentences: int = 3) -> str:
    """Summarize a long piece of text.

    Args:
        text: The text to summarize.
        max_sentences: Target maximum number of sentences in the summary.
    """
    llm = chat_llm(temperature=0.2)
    system = SystemMessage(content=(
        f"Summarize the following text in at most {max_sentences} sentences. "
        "Output only the summary."
    ))
    result = invoke_with_retry(llm, [system, HumanMessage(content=text)])
    return result.content


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Prepare an email to send. NOTE: this does NOT actually send email —
    no SMTP/OAuth is configured in this build. It returns a draft for the
    user to review and send manually, and is treated as a sensitive
    action requiring human approval before even that.

    Args:
        to: Recipient.
        subject: Email subject line.
        body: Email body.
    """
    # Honest behavior: never claim to have sent something we didn't send.
    return (
        f"DRAFT ONLY (not sent — no email service is connected in this build):\n"
        f"To: {to}\nSubject: {subject}\n\n{body}"
    )


COMMUNICATION_TOOLS = [draft_message, rewrite_professionally, summarize_text, send_email]
