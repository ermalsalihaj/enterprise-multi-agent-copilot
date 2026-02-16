"""LangGraph workflow: Plan → Research → Write → Verify → Deliver."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from .state import GraphState
from .planner import planner_node
from .researcher import researcher_node
from .writer import writer_node
from .verifier import verifier_node


def build_workflow():
    """
    Build the LangGraph workflow implementing:
    Plan → Research → Draft → Verify → Deliver
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("verifier", verifier_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "verifier")
    workflow.add_edge("verifier", END)

    graph = workflow.compile()
    return graph


def _build_observability(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_agent = []
    total_latency_ms = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_errors = 0

    for event in trace:
        output = event.get("output", {}) or {}
        usage = output.get("token_usage", {}) or {}
        latency = int(output.get("latency_ms", 0) or 0)
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
        errors = int(output.get("errors", 0) or 0)

        total_latency_ms += latency
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_tokens += tokens
        total_errors += errors

        per_agent.append(
            {
                "agent": event.get("agent", "unknown"),
                "latency_ms": latency,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": tokens,
                "errors": errors,
            }
        )

    return {
        "per_agent": per_agent,
        "totals": {
            "latency_ms": total_latency_ms,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "errors": total_errors,
        },
    }


def run_copilot(
    question: str,
    goal: str,
    output_mode: str = "executive",
    email_signer: str = "",
) -> Dict[str, Any]:
    """Run the full workflow and return verified_output, trace, and observability."""
    graph = build_workflow()
    initial: Dict[str, Any] = {
        "question": question,
        "goal": goal,
        "output_mode": output_mode,
        "email_signer": (email_signer or "").strip(),
        "trace": [],
    }
    result = graph.invoke(initial)
    trace = result.get("trace", [])
    observability = _build_observability(trace)
    return {
        "verified_output": result.get("verified_output", {}),
        "trace": trace,
        "observability": observability,
    }
