"""
Evaluation script. Run with:

    python scripts/evaluate.py

Requires GROQ_API_KEY to be set (it runs the real graph against every
case in the eval dataset — there's no way to measure real LLM behavior
without calling the real LLM). If no key is set, this prints what it
would do instead of failing silently.

Metrics computed:
    - Routing accuracy       : did the router pick the expected intent?
    - Tool selection accuracy: for task/communication cases, was the
                                expected tool actually called?
    - Retrieval quality      : for knowledge cases with expected keywords,
                                did the answer contain them (proxy for
                                faithfulness on a small hand-labeled set)?
    - Correct refusal rate   : for knowledge cases with NO expected
                                keywords (unanswerable from the KB), did
                                the system correctly refuse instead of
                                hallucinating?
    - Approval-gating accuracy: did sensitive actions correctly require
                                approval instead of auto-executing?
    - Latency / token usage  : per-case and averaged
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.config.settings import settings
from backend.evaluation.eval_dataset import EVAL_DATASET


def check_api_key():
    if not settings.GROQ_API_KEY:
        print("=" * 70)
        print("GROQ_API_KEY is not set — cannot run live evaluation.")
        print("Add your key to .env, then re-run: python scripts/evaluate.py")
        print()
        print(f"This would evaluate {len(EVAL_DATASET)} cases across categories:")
        categories = {}
        for case in EVAL_DATASET:
            categories[case.category] = categories.get(case.category, 0) + 1
        for cat, count in categories.items():
            print(f"  - {cat}: {count} cases")
        print("=" * 70)
        return False
    return True


def run_evaluation():
    if not check_api_key():
        return

    from backend.graph.graph import max_graph

    results = []
    print(f"Running evaluation on {len(EVAL_DATASET)} cases...\n")

    for case in EVAL_DATASET:
        start = time.perf_counter()
        try:
            state = max_graph.invoke({
                "user_input": case.input,
                "session_id": f"eval_{case.id}",
                "conversation_history": [],
                "llm_call_count": 0,
                "token_estimate": 0,
                "tools_used": [],
                "sources": [],
            })
            error = None
        except Exception as e:
            state = {}
            error = str(e)

        latency = time.perf_counter() - start

        intent_correct = state.get("intent") == case.expected_intent
        tool_correct = (
            case.expected_tool is None
            or case.expected_tool in state.get("tools_used", [])
            or (case.requires_approval_expected and state.get("requires_approval"))
        )
        approval_correct = state.get("requires_approval", False) == case.requires_approval_expected

        keyword_hit = None
        if case.category == "knowledge":
            response_lower = state.get("response", "").lower()
            if case.expected_keywords:
                keyword_hit = any(kw.lower() in response_lower for kw in case.expected_keywords)
            else:
                # Should correctly refuse rather than hallucinate an answer
                keyword_hit = "don't have enough evidence" in response_lower or "don't have any documents" in response_lower

        results.append({
            "id": case.id,
            "category": case.category,
            "error": error,
            "intent_correct": intent_correct,
            "tool_correct": tool_correct,
            "approval_correct": approval_correct,
            "keyword_check": keyword_hit,
            "latency_seconds": round(latency, 2),
            "llm_calls": state.get("llm_call_count", 0),
            "tokens": state.get("token_estimate", 0),
            "response_preview": (state.get("response", "") or "")[:120],
        })

        status = "OK" if error is None else f"ERROR: {error}"
        print(f"[{case.id}] {status} | intent={state.get('intent')} (expected {case.expected_intent}) | {latency:.2f}s")

    print_summary(results)
    return results


def print_summary(results):
    total = len(results)
    errors = sum(1 for r in results if r["error"])
    intent_acc = sum(1 for r in results if r["intent_correct"]) / total
    tool_acc = sum(1 for r in results if r["tool_correct"]) / total
    approval_acc = sum(1 for r in results if r["approval_correct"]) / total

    knowledge_results = [r for r in results if r["category"] == "knowledge"]
    knowledge_acc = (
        sum(1 for r in knowledge_results if r["keyword_check"]) / len(knowledge_results)
        if knowledge_results else None
    )

    avg_latency = sum(r["latency_seconds"] for r in results) / total
    avg_tokens = sum(r["tokens"] for r in results) / total
    total_llm_calls = sum(r["llm_calls"] for r in results)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total cases:              {total}")
    print(f"Errors:                   {errors}")
    print(f"Routing accuracy:         {intent_acc:.1%}")
    print(f"Tool selection accuracy:  {tool_acc:.1%}")
    print(f"Approval-gating accuracy: {approval_acc:.1%}")
    if knowledge_acc is not None:
        print(f"Retrieval/faithfulness:   {knowledge_acc:.1%}  (keyword-match proxy on {len(knowledge_results)} cases)")
    print(f"Avg latency:              {avg_latency:.2f}s")
    print(f"Avg tokens (estimated):   {avg_tokens:.0f}")
    print(f"Total LLM calls:          {total_llm_calls}")
    print("=" * 70)

    os.makedirs("./data", exist_ok=True)
    with open("./data/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to ./data/eval_results.json")


if __name__ == "__main__":
    run_evaluation()
