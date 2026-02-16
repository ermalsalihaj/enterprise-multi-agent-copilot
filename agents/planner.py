"""Planner agent: decomposes the task and produces a plan."""
from __future__ import annotations

import time
from typing import Any

from config import settings

from .llm import invoke_openai_chat
from .state import GraphState, PROMPT_INJECTION_DEFENSE


def planner_node(state: GraphState) -> dict[str, Any]:
    """Create a structured plan from the business question and goal."""
    system = (
        PROMPT_INJECTION_DEFENSE + " "
        "You are a strategic planner for insurance operations. "
        "Given a business question and goal, produce a clear, step-by-step plan for research and delivery. "
        "Output only the plan text, no preamble."
    )
    user = (
        f"Business question: {state['question']}\n\n"
        f"Goal: {state['goal']}\n\n"
        "Provide a concise plan (bullet points or short paragraphs)."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    start = time.perf_counter()
    errors = 0
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        plan, token_usage = invoke_openai_chat(
            settings.model_main,
            settings.openai_api_key,
            messages,
            temperature=0.2,
        )
        if not plan:
            plan = "No plan generated."
    except Exception as e:
        plan = f"Plan generation failed: {e}"
        errors = 1
    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "plan": plan,
        "trace": [
            {
                "agent": "planner",
                "notes": "Decomposed task into plan",
                "input": {"question": state["question"], "goal": state["goal"]},
                "output": {
                    "plan": plan,
                    "latency_ms": latency_ms,
                    "token_usage": token_usage,
                    "errors": errors,
                },
            }
        ],
    }
