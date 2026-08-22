# ARCHITECTURE.md

Deep-dive on every component, why it exists, and the trade-offs behind it. Written for interview prep as much as onboarding — if you can defend every decision on this page, you can defend this project.

---

## 1. Graph Structure

```
router -> memory_agent -> {chat_agent | task_agent | rag_agent}
       -> verification -> {guardrail_block | memory_persist} -> END
```

Two early-exit paths exist:
1. **Sensitive action pending** — a specialist agent sets `requires_approval=True`, the graph ends immediately (before verification), and the API layer surfaces the pending action to the frontend. A second `/approve` call handles the human's decision outside the graph.
2. **Guardrail block** — verification flags the response unsafe, and it's replaced before reaching the user.

### Why not run human approval *inside* the graph with a blocking wait?

LangGraph nodes are synchronous functions in this build — blocking one on a human clicking a button in a browser would tie up a request thread indefinitely and doesn't map cleanly onto a stateless HTTP API. Ending the graph and re-entering via a second endpoint call keeps the graph itself simple, testable, and non-blocking, at the cost of the pending-approval state living in a process-local dict (see "Known Limitations" below).

---

## 2. Node-by-Node

### `router_node`
Uses the **fast model** (Llama 3.1 8B) with structured output (`IntentClassification` Pydantic schema) to classify intent into `chat | task | knowledge | communication`. Fails open to `chat` on any error — a router crash should never crash the whole assistant.

**Why structured output instead of parsing free text?** Reliability. A model asked to "respond with one word" can drift ("Sure, I'd say this is a 'task'!"); a Pydantic schema enforced via `with_structured_output` can't.

### `memory_agent_node`
Runs a lightweight keyword search against long-term memory + pulls all preferences, before the specialist agent runs. Kept as its own node (not folded into each specialist) so every agent gets the same memory-injection behavior for free.

### `chat_agent_node` / `task_agent_node` / `rag_agent_node`
The three specialists. `task_agent_node` is the most complex: it binds tools to the LLM, checks the requested tool(s) against `settings.SENSITIVE_TOOLS` **before** executing anything, and only proceeds to a second LLM call (composing the final reply) after tool execution.

### `verification_node`
Checks groundedness + safety via a structured-output call to the fast model — **but skips itself entirely** for short chat responses (`intent == "chat"` and `len(response) < 300`). This is a real latency/cost optimization: verifying "Hey, how's it going? Doing well, thanks for asking!" adds a full LLM round-trip for zero practical safety benefit.

### `guardrail_block_node`
Replaces an unsafe response with a refusal, explaining (without violating the point of the block) that a safety concern was flagged.

### `maybe_persist_memory_node`
Keyword-triggered (`"remember"`, `"my name is"`, `"i prefer"`, etc.) — deliberately simple and conservative. **This directly implements the "do not store everything automatically" requirement.**

---

## 3. RAG Pipeline Deep-Dive

```
query
  -> hybrid_search()      [BM25 top-k + vector top-k, merged with 60/40 weighting]
  -> rerank()              [cross-encoder/ms-marco-MiniLM-L-6-v2, lazy-loaded]
  -> retrieval_confidence() [squashed top reranker score -> 0-1]
  -> if confidence < 0.35: honest refusal
  -> else: answer with citation, using only retrieved context
```

**Why 60/40 vector/keyword weighting instead of 50/50 or a learned weight?** A practical starting point, not a tuned constant — vector search is generally the stronger signal for semantic questions, keyword search is there specifically to catch what vector search misses (exact numbers, names). A real production system would tune this weight against a labeled retrieval-quality benchmark; this build's `MIN_RETRIEVAL_CONFIDENCE` and merge weights are both configurable via `.env` for exactly that reason.

**Why chunk with sentence-boundary snapping instead of fixed-size?** Fixed-size chunking regularly cuts a sentence in half, which both hurts embedding quality (a half-sentence embeds worse than a complete one) and looks unprofessional in a cited answer.

**Known limitation:** the confidence score (`(top_score + 5) / 10`) is an empirical squash calibrated by hand against the ms-marco-MiniLM score range observed during testing, not a statistically calibrated probability. It's honestly labeled as such in the code — a genuinely calibrated confidence score (e.g. via isotonic regression against a labeled relevant/irrelevant dataset) is listed under Future Improvements in the README.

---

## 4. Memory Architecture Deep-Dive

| Store | Persistence | Written when | Deletable |
|---|---|---|---|
| Short-term | In-process only | Every turn | Cleared per-session |
| Long-term | SQLite | Keyword-triggered match | Yes, via API/frontend |
| Preferences | SQLite | Explicit `/memory/preferences` call | Yes, via API/frontend |

The keyword-trigger list for long-term memory (`MEMORY_WORTHY_KEYWORDS` in `nodes.py`) is intentionally short and conservative. A more sophisticated version would use a small classifier or a dedicated LLM call to judge memory-worthiness — that's a reasonable v2, traded off here against keeping the persistence decision fast, cheap (zero extra LLM calls), and easy to audit/explain in an interview.

---

## 5. Observability Deep-Dive

Two tiers:
1. **LangSmith** — activates automatically via env vars (`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`) since LangChain's SDK reads these natively. Zero code changes needed to turn it on.
2. **Local SQLite tracer** (`backend/services/observability.py`) — every node call is wrapped in `trace_node()`, logging latency + arbitrary metadata. This is what powers the frontend's System Status page and answers "which node is slow" even with zero external accounts configured.

**Why build a local tracer at all, if LangSmith exists?** Because a portfolio project that only works once a third-party account is configured demonstrates less than one that works immediately and *also* integrates with the industry-standard tool. It also means the `/traces/*` API endpoints and the frontend's observability panel have real data to show in a demo/interview setting without requiring the interviewer to see a LangSmith dashboard.

---

## 6. Evaluation Methodology

`backend/evaluation/eval_dataset.py` — 14 hand-written cases across chat/task/knowledge/communication, each with an *expected*, checkable outcome (expected intent, expected tool, expected keywords, or expected approval-gating behavior).

`scripts/evaluate.py` runs the real graph against every case and computes:
- **Routing accuracy** — did the router pick the right intent?
- **Tool selection accuracy** — was the right tool actually called?
- **Approval-gating accuracy** — did sensitive actions correctly require approval?
- **Retrieval/faithfulness (keyword-match proxy)** — for knowledge cases with expected keywords, did the answer contain them; for the one deliberately-unanswerable case, did the system correctly refuse instead of hallucinate?

**Honest limitation:** keyword-match is a proxy for faithfulness, not a full semantic entailment check. A more rigorous version would use an LLM-as-judge to verify every claim in the response against the retrieved context sentence-by-sentence — a natural extension once the eval dataset grows past hand-checkable size.

---

## 7. Reliability Mechanisms

- **Retries with backoff** — `invoke_with_retry()` in `llm_service.py`, used by every LLM call in the graph
- **Timeouts** — every `ChatGroq` instance is constructed with `settings.LLM_TIMEOUT_SECONDS`
- **Fail-open router** — a router failure defaults to `chat` rather than crashing
- **Fail-open verification** — a verification failure defaults to `is_safe=True, is_grounded=True` but logs `verification_check_failed` as an issue, so it's visible in traces without blocking the user
- **Honest tool failure** — `send_email` never claims to have sent anything; unimplemented actions say so explicitly rather than pretending

---

## 8. Known Limitations (stated directly, not hidden)

- Pending-approval state (`_pending_approvals` in `routes.py`) is a process-local dict — resets on server restart, and won't work correctly across multiple backend instances behind a load balancer. A real deployment needs this in Redis or the database.
- Long-term memory's "is this worth remembering" decision is keyword-based, not model-based — simple and auditable, but will miss memory-worthy statements that don't match the keyword list.
- Retrieval confidence is an empirical heuristic, not a calibrated probability (see RAG section above).
- `send_email` is a stub by design — no SMTP/OAuth integration exists yet, and it says so honestly rather than faking success.
- Document ingestion supports `.txt`/`.md` only — PDF/DOCX need an extraction step not built yet.
- The evaluation script's faithfulness check is a keyword-match proxy, not full semantic verification.

Every one of these is a legitimate "what would you do next" answer in an interview — they're listed here specifically so the answer is ready.
