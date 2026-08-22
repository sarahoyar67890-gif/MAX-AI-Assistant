# MAX AI Assistant — Advanced

A **Personal AI Operations Assistant** built on a multi-agent LangGraph orchestration layer, with advanced hybrid RAG, persistent memory, real tool calling, human-in-the-loop approval for sensitive actions, an evaluation harness, and a local observability layer.

This is not a single-prompt chatbot with a router bolted on. Every request flows through intent classification → context retrieval (memory + knowledge base) → a specialist agent → verification → (approval gate, if needed) → response — with every node's latency and decisions logged for inspection.

---

## Problem Statement

Most personal AI assistant demos are a single LLM call with a system prompt. They can't:
- Tell you *why* they answered the way they did
- Refuse to answer when they don't actually know something
- Ask before taking a sensitive action
- Remember anything across sessions in a controlled, inspectable way
- Show you where they're slow or expensive to run

MAX is built to do all five, using techniques (LangGraph orchestration, hybrid retrieval + reranking, confidence-gated RAG, human-in-the-loop, structured evaluation) that map directly to how production AI systems are actually engineered — not just demoed.

---

## Features

- **Multi-agent orchestration (LangGraph)** — router classifies intent, hands off to a chat / task / RAG specialist agent
- **Advanced RAG** — hybrid retrieval (BM25 keyword + dense vector search) + cross-encoder reranking + confidence gating (refuses to answer rather than hallucinate when evidence is weak)
- **Real tool calling** — productivity tools (tasks, prioritization, checklists) and communication tools (draft, rewrite, summarize) that actually execute, and honestly report when they can't
- **Human-in-the-loop** — sensitive actions (e.g. sending email) require explicit approval before executing, not after
- **Three-tier memory** — short-term (conversation window), long-term (persisted, explicit, inspectable/deletable), and preferences (behavior instructions), never written automatically
- **Verification & guardrails** — a dedicated node checks groundedness and safety before a response goes out
- **Observability** — every node's latency/metadata logged locally (SQLite) with zero setup; LangSmith tracing activates automatically if you add a key
- **Evaluation harness** — a hand-built dataset + `scripts/evaluate.py` scoring routing accuracy, tool selection accuracy, retrieval faithfulness, and approval-gating correctness
- **Latency/cost optimization** — tiered models (cheap/fast model for routing & classification, full model for generation), a heuristic skip on verification for short chat responses, lazy-loaded reranker
- **Premium dashboard frontend** — chat, knowledge base management, memory inspection, task view, and a live system-status/observability panel

---

## Architecture

```
User Input
    |
    v
Router Node (fast model, structured output classification)
    |
    v
Memory Agent (pulls relevant long-term memory + preferences)
    |
    +---------------+---------------+
    |               |               |
Chat Agent      Task Agent      RAG Agent
(conversation)  (tool calling,   (hybrid retrieval +
                 sensitive-      reranking + confidence
                 action gating)  gating)
    |               |               |
    +---------------+---------------+
                    |
        [ if sensitive action pending -> END, await human approval ]
                    |
                    v
          Verification Node
      (groundedness + safety check,
       skipped for short chat replies
       as a latency optimization)
                    |
          +---------+---------+
          |                   |
   Guardrail Block      Finalize
   (unsafe response       |
    replaced)              v
          |          Memory Persist
          +---------------+
                    |
                   END
```

See `ARCHITECTURE.md` for the full breakdown of every node, why it exists, and the specific engineering trade-offs behind it.

### Why LangGraph, and why multiple agents

A single LLM call can't conditionally branch, can't gate on a confidence score, and can't cleanly separate "decide what to do" from "do it." LangGraph's explicit state machine makes each of those a first-class node with its own tracing, testability, and failure-handling — which is also *why* this project has 69 passing tests: each node's logic is isolated enough to unit test with mocked LLM calls, instead of only being testable end-to-end.

### RAG architecture

`chunk (sentence-aware, overlapping) → hybrid retrieval (BM25 + vector, weighted merge) → cross-encoder rerank → confidence score → answer or honest refusal`

Hybrid retrieval exists because pure vector similarity misses exact matches (dollar amounts, names, IDs) that keyword search catches trivially. Reranking exists because embedding similarity and true relevance aren't the same thing — a cross-encoder that actually reads the query against each candidate reorders more accurately than cosine similarity alone. Confidence gating exists so MAX says "I don't have enough evidence" instead of confidently making something up when retrieval is weak.

### Memory architecture

Three stores, kept separate on purpose:
- **Short-term** — in-process, per-session, trimmed conversation window. Never touches disk.
- **Long-term** — SQLite-backed, only written when a message matches memory-worthy patterns (not everything, automatically) — inspectable and deletable via the API/frontend.
- **Preferences** — separate from long-term memory because preferences change MAX's *behavior*, not its knowledge.

### How hallucinations are reduced

Three independent mechanisms, not one: (1) RAG confidence gating refuses to answer from weak retrieval, (2) the task agent is explicitly instructed to say when it can't perform an action rather than claim it did, (3) the verification node does a final groundedness pass on the response before it's returned.

### How latency/cost were reduced

- **Tiered models** — routing/classification/verification use a small fast model; only response generation uses the full model
- **Verification short-circuit** — short chat responses skip the LLM verification call entirely (a real, measurable optimization, not just a fallback)
- **Lazy-loaded reranker** — the cross-encoder model only loads on first actual use, not at startup
- **BM25 rebuilt in-memory** rather than per-query disk reads

---

## Tech Stack

Python · FastAPI · LangGraph · LangChain · Groq (Llama 3.3 70B + Llama 3.1 8B) · ChromaDB · Sentence-Transformers (embeddings + cross-encoder reranking) · rank_bm25 · SQLite · Streamlit · Docker · pytest

---

## Installation

```bash
git clone <your-repo-url>
cd MAX-AI-ASSISTANT-ADVANCED
pip install -r requirements.txt
cp .env.example .env
# then edit .env and add your GROQ_API_KEY
```

Get a free Groq API key at https://console.groq.com

## Environment Variables

See `.env.example` for the full list with defaults. The only **required** variable is `GROQ_API_KEY`. Everything else (model choice, RAG tuning, storage paths, LangSmith) has a sensible default.

## Running Locally

**Backend:**
```bash
uvicorn main:app --reload --port 8000
```

**Frontend** (separate terminal):
```bash
streamlit run frontend/app.py
```

Then open http://localhost:8501

## Running with Docker

```bash
docker compose up --build
```

Backend: http://localhost:8000 · Frontend: http://localhost:8501

> Note: the Docker setup follows standard patterns and every dependency installs cleanly, but `docker build` itself wasn't run in the environment this was built in (no Docker daemon available there) — run `docker compose up --build` yourself and check the logs on first run.

## Running Tests

```bash
pytest backend/tests/ -v
```

All 69 tests should pass. They run fully offline — LLM calls and embedding models are mocked so tests don't need your Groq key or network access. This was a deliberate choice made while building this: tests should be fast, deterministic, and runnable in CI without external dependencies.

## Running the Evaluation Script

```bash
python scripts/evaluate.py
```

Requires `GROQ_API_KEY` to be set — it runs the real graph against every case in `backend/evaluation/eval_dataset.py`. Without a key, it prints what it would evaluate instead of failing. Results are saved to `data/eval_results.json`.

**To evaluate the RAG/knowledge cases properly**, first upload `data/sample_docs/company_policy.txt` via the frontend's Knowledge Base page (or `POST /documents/upload`) so the knowledge-base test cases have something real to retrieve from.

---

## API Documentation

Full interactive docs at `http://localhost:8000/docs` once the backend is running (FastAPI's auto-generated Swagger UI).

Key endpoints:
| Endpoint | Purpose |
|---|---|
| `POST /chat` | Send a message, get MAX's response + performance info |
| `POST /approve` | Approve/reject a pending sensitive action |
| `POST /documents/upload` | Index a .txt/.md file into the knowledge base |
| `GET /memory/long-term` | List stored long-term memories |
| `DELETE /memory/long-term/{id}` | Delete a memory |
| `GET /traces/performance-summary` | Per-node average latency (observability) |

---

## Example Use Cases

- **"Remind me to submit my assignment tonight"** → task agent → `create_task` tool → confirmed
- **"How much is the home office stipend?"** (after uploading the sample policy doc) → RAG agent → hybrid retrieval + rerank → cited answer
- **"What's the capital of Australia?"** (not in your knowledge base) → RAG agent → honest "I don't have enough evidence" refusal instead of a guess
- **"Draft an email to my manager asking for Friday off"** → task agent → drafts it, then **requires your approval** before treating it as sent (and even then, only prepares a draft — no live email service is wired up)

---

## Future Improvements

Being direct about what's next, not what's already claimed as done:
- PDF/DOCX ingestion (currently .txt/.md only)
- Real email sending (SMTP/OAuth) behind the existing approval gate
- Persistent pending-approvals store (currently process-local, resets on restart)
- Query rewriting / multi-query retrieval for harder RAG questions
- A calibrated (not heuristic) retrieval confidence score
- Redis-backed short-term memory for multi-instance deployments
- CI pipeline (GitHub Actions) running the test suite on every push

---

## Screenshots

*(Add screenshots of the Chat, Knowledge Base, Memory, and System Status pages here once you've run it — a portfolio README is stronger with real screenshots than a placeholder.)*
