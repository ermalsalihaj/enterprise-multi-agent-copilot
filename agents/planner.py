"""Planner agent: decomposes the task and produces a plan."""
from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from .state import GraphState


def planner_node(state: GraphState) -> dict[str, Any]:
    """Create a structured plan from the business question and goal."""
    llm = ChatOpenAI(
        model=settings.model_main,
        api_key=settings.openai_api_key,
        temperature=0.2,
    )
    system = (
        "You are a strategic planner for insurance operations. "
        "Given a business question and goal, produce a clear, step-by-step plan for research and delivery. "
        "Output only the plan text, no preamble."
    )
    user = (
        f"Business question: {state['question']}\n\n"
        f"Goal: {state['goal']}\n\n"
        "Provide a concise plan (bullet points or short paragraphs)."
    )
    start = time.perf_counter()
    errors = 0
    response = None
    try:
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        plan = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        plan = f"Plan generation failed: {e}"
        errors = 1
    latency_ms = int((time.perf_counter() - start) * 1000)
    usage = (getattr(response, "response_metadata", {}) or {}) if response else {}
    usage = usage.get("usage", {}) or {}
    token_usage = {
        "prompt_tokens": usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0) or usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }
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
