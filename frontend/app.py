"""
MAX AI Assistant — Premium Frontend

Run with:
    streamlit run frontend/app.py

Talks to the FastAPI backend (default http://localhost:8000). This is
built as a heavily custom-styled Streamlit app rather than a separate
React app — no npm/build pipeline needed, runs with one command, and
still gets a real dashboard layout (sidebar nav, cards, status panels)
instead of the default "chat box on a blank page" look most Streamlit
demos have.
"""

import streamlit as st
import requests
import uuid
import os
from datetime import datetime

# Reads from env var so Docker Compose can point this at the "backend"
# service name instead of localhost — falls back to localhost for local dev.
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="MAX AI Assistant", page_icon="◆", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Custom CSS — dark, modern, "AI product" feel instead of default Streamlit
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background-color: #0b0d12; }

    section[data-testid="stSidebar"] {
        background-color: #12151c;
        border-right: 1px solid #1f2430;
    }

    .max-card {
        background: #12151c;
        border: 1px solid #1f2430;
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 14px;
    }

    .max-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f4f5f7;
        letter-spacing: -0.3px;
    }

    .max-subtitle {
        color: #8b93a7;
        font-size: 0.85rem;
        margin-top: -6px;
    }

    .status-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-online { background: rgba(52, 211, 153, 0.15); color: #34d399; }
    .status-offline { background: rgba(248, 113, 113, 0.15); color: #f87171; }

    .metric-box {
        background: #171a22;
        border: 1px solid #1f2430;
        border-radius: 10px;
        padding: 14px 16px;
        text-align: center;
    }
    .metric-value { font-size: 1.3rem; font-weight: 700; color: #f4f5f7; }
    .metric-label { font-size: 0.72rem; color: #8b93a7; text-transform: uppercase; letter-spacing: 0.5px; }

    .chat-bubble-user {
        background: #2563eb;
        color: white;
        padding: 10px 16px;
        border-radius: 14px 14px 2px 14px;
        max-width: 75%;
        margin-left: auto;
        margin-bottom: 10px;
    }
    .chat-bubble-max {
        background: #171a22;
        border: 1px solid #1f2430;
        color: #e5e7eb;
        padding: 10px 16px;
        border-radius: 14px 14px 14px 2px;
        max-width: 75%;
        margin-bottom: 10px;
    }
    .chat-meta { font-size: 0.7rem; color: #6b7280; margin-top: 2px; }

    .approval-box {
        background: rgba(251, 191, 36, 0.08);
        border: 1px solid rgba(251, 191, 36, 0.3);
        border-radius: 10px;
        padding: 14px 16px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None


def check_backend_online():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.status_code == 200, r.json()
    except requests.exceptions.RequestException:
        return False, {}


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="max-title">◆ MAX</div>', unsafe_allow_html=True)
    st.markdown('<div class="max-subtitle">Personal AI Operations Assistant</div>', unsafe_allow_html=True)
    st.markdown("---")

    online, health = check_backend_online()
    pill_class = "status-online" if online else "status-offline"
    pill_text = "● Online" if online else "● Offline"
    st.markdown(f'<span class="status-pill {pill_class}">{pill_text}</span>', unsafe_allow_html=True)
    if online:
        st.caption(f"Knowledge base: {health.get('knowledge_base_documents', 0)} chunks indexed")
        st.caption(f"LangSmith tracing: {'enabled' if health.get('langsmith_enabled') else 'not configured'}")
    else:
        st.caption("Start the backend: `uvicorn main:app --reload --port 8000`")

    st.markdown("---")
    page = st.radio("Navigate", ["Chat", "Knowledge Base", "Memory", "Tasks", "System Status"], label_visibility="collapsed")


# ---------------------------------------------------------------------------
# CHAT PAGE
# ---------------------------------------------------------------------------
if page == "Chat":
    st.markdown('<div class="max-title">Chat with MAX</div>', unsafe_allow_html=True)
    st.markdown('<div class="max-subtitle">Ask anything — MAX routes to the right agent automatically.</div>', unsafe_allow_html=True)
    st.write("")

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-max">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("perf"):
                p = msg["perf"]
                st.markdown(
                    f'<div class="chat-meta">route: {p.get("route")} · '
                    f'{p.get("latency_seconds")}s · {p.get("llm_calls")} LLM call(s) · '
                    f'~{p.get("tokens_used")} tokens'
                    + (f' · tools: {", ".join(p.get("tools_used", []))}' if p.get("tools_used") else '')
                    + '</div>',
                    unsafe_allow_html=True,
                )
            if msg.get("sources"):
                st.caption(f"Sources: {', '.join(msg['sources'])}")

    if st.session_state.pending_approval:
        st.markdown(
            f'<div class="approval-box">⚠ <b>Approval needed:</b> {st.session_state.pending_approval}</div>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        if col1.button("✓ Approve", use_container_width=True):
            try:
                requests.post(f"{API_BASE}/approve", json={"session_id": st.session_state.session_id, "approved": True}, timeout=10)
            except requests.exceptions.RequestException:
                pass
            st.session_state.pending_approval = None
            st.rerun()
        if col2.button("✗ Reject", use_container_width=True):
            try:
                requests.post(f"{API_BASE}/approve", json={"session_id": st.session_state.session_id, "approved": False}, timeout=10)
            except requests.exceptions.RequestException:
                pass
            st.session_state.pending_approval = None
            st.rerun()

    user_input = st.chat_input("Message MAX...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        try:
            r = requests.post(
                f"{API_BASE}/chat",
                json={"message": user_input, "session_id": st.session_state.session_id},
                timeout=30,
            )
            if r.status_code == 200:
                body = r.json()
                st.session_state.messages.append({
                    "role": "assistant", "content": body["response"],
                    "perf": body.get("performance"), "sources": body.get("sources", []),
                })
                if body.get("requires_approval"):
                    st.session_state.pending_approval = body.get("approval_action")
            else:
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {r.json().get('detail', 'unknown error')}"})
        except requests.exceptions.RequestException as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Couldn't reach the backend: {e}"})
        st.rerun()


# ---------------------------------------------------------------------------
# KNOWLEDGE BASE PAGE
# ---------------------------------------------------------------------------
elif page == "Knowledge Base":
    st.markdown('<div class="max-title">Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown('<div class="max-subtitle">Upload documents for MAX to answer questions from (RAG).</div>', unsafe_allow_html=True)
    st.write("")

    uploaded = st.file_uploader("Upload a .txt or .md file", type=["txt", "md"])
    if uploaded and st.button("Index this document"):
        try:
            files = {"file": (uploaded.name, uploaded.getvalue(), "text/plain")}
            r = requests.post(f"{API_BASE}/documents/upload", files=files, timeout=30)
            if r.status_code == 200:
                st.success(f"Indexed {uploaded.name} — {r.json()['chunks_created']} chunks created.")
            else:
                st.error(r.json().get("detail", "Upload failed."))
        except requests.exceptions.RequestException as e:
            st.error(f"Couldn't reach the backend: {e}")

    st.markdown("---")
    try:
        r = requests.get(f"{API_BASE}/documents", timeout=5)
        if r.status_code == 200:
            data = r.json()
            st.markdown('<div class="max-card">', unsafe_allow_html=True)
            st.write(f"**Total chunks indexed:** {data['total_chunks']}")
            st.write("**Sources:**")
            for s in data["sources"]:
                st.write(f"- {s}")
            st.markdown('</div>', unsafe_allow_html=True)
    except requests.exceptions.RequestException:
        st.info("Backend not reachable — start it to see indexed documents.")


# ---------------------------------------------------------------------------
# MEMORY PAGE
# ---------------------------------------------------------------------------
elif page == "Memory":
    st.markdown('<div class="max-title">Memory</div>', unsafe_allow_html=True)
    st.markdown('<div class="max-subtitle">What MAX remembers about you — inspect and delete anytime.</div>', unsafe_allow_html=True)
    st.write("")

    tab1, tab2 = st.tabs(["Long-term memory", "Preferences"])

    with tab1:
        try:
            r = requests.get(f"{API_BASE}/memory/long-term", timeout=5)
            memories = r.json().get("memories", []) if r.status_code == 200 else []
        except requests.exceptions.RequestException:
            memories = []
            st.info("Backend not reachable.")

        if not memories:
            st.caption("No long-term memories stored yet.")
        for m in memories:
            col1, col2 = st.columns([5, 1])
            col1.markdown(f'<div class="max-card">{m["content"]}<div class="chat-meta">{m["created_at"]}</div></div>', unsafe_allow_html=True)
            if col2.button("Delete", key=f"del_mem_{m['id']}"):
                requests.delete(f"{API_BASE}/memory/long-term/{m['id']}", timeout=5)
                st.rerun()

    with tab2:
        new_pref = st.text_input("Add a preference (e.g. 'Keep answers short')")
        if st.button("Add preference") and new_pref:
            requests.post(f"{API_BASE}/memory/preferences", params={"content": new_pref}, timeout=5)
            st.rerun()

        try:
            r = requests.get(f"{API_BASE}/memory/preferences", timeout=5)
            prefs = r.json().get("preferences", []) if r.status_code == 200 else []
        except requests.exceptions.RequestException:
            prefs = []

        for p in prefs:
            col1, col2 = st.columns([5, 1])
            col1.markdown(f'<div class="max-card">{p["content"]}</div>', unsafe_allow_html=True)
            if col2.button("Delete", key=f"del_pref_{p['id']}"):
                requests.delete(f"{API_BASE}/memory/preferences/{p['id']}", timeout=5)
                st.rerun()


# ---------------------------------------------------------------------------
# TASKS PAGE
# ---------------------------------------------------------------------------
elif page == "Tasks":
    st.markdown('<div class="max-title">Tasks</div>', unsafe_allow_html=True)
    st.markdown('<div class="max-subtitle">Tasks MAX has created for you via chat.</div>', unsafe_allow_html=True)
    st.write("")

    try:
        r = requests.get(f"{API_BASE}/tasks", timeout=5)
        tasks_text = r.json().get("tasks", "No tasks found.") if r.status_code == 200 else "Backend not reachable."
    except requests.exceptions.RequestException:
        tasks_text = "Backend not reachable."

    st.markdown(f'<div class="max-card"><pre style="color:#e5e7eb; white-space: pre-wrap;">{tasks_text}</pre></div>', unsafe_allow_html=True)
    st.caption("Create tasks by asking MAX in chat, e.g. \"remind me to submit my assignment tonight\".")


# ---------------------------------------------------------------------------
# SYSTEM STATUS PAGE (observability)
# ---------------------------------------------------------------------------
elif page == "System Status":
    st.markdown('<div class="max-title">System Status</div>', unsafe_allow_html=True)
    st.markdown('<div class="max-subtitle">Observability — node performance and recent traces.</div>', unsafe_allow_html=True)
    st.write("")

    try:
        r = requests.get(f"{API_BASE}/traces/performance-summary", timeout=5)
        perf = r.json().get("node_performance", []) if r.status_code == 200 else []
    except requests.exceptions.RequestException:
        perf = []
        st.info("Backend not reachable.")

    if perf:
        cols = st.columns(len(perf))
        for col, node in zip(cols, perf):
            with col:
                st.markdown(
                    f'<div class="metric-box"><div class="metric-value">{node["avg_latency_ms"]:.0f}ms</div>'
                    f'<div class="metric-label">{node["node_name"]}</div></div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"{node['call_count']} calls · max {node['max_latency_ms']:.0f}ms")
    else:
        st.caption("No trace data yet — send a few chat messages first.")

    st.markdown("---")
    st.markdown("**Recent traces**")
    try:
        r = requests.get(f"{API_BASE}/traces/recent", params={"limit": 20}, timeout=5)
        traces = r.json().get("traces", []) if r.status_code == 200 else []
        if traces:
            st.dataframe(
                [{"node": t["node_name"], "latency_ms": round(t["latency_ms"], 1), "session": t["session_id"], "time": t["created_at"]} for t in traces],
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("No traces recorded yet.")
    except requests.exceptions.RequestException:
        pass
