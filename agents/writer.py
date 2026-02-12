"""Writer agent: produces structured deliverable (executive summary, email, action list)."""
from __future__ import annotations

import json
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings

from .state import GraphState


def writer_node(state: GraphState) -> dict[str, Any]:
    """Produce draft deliverable from plan and research notes."""
    llm = ChatOpenAI(
        model=settings.model_main,
        api_key=settings.openai_api_key,
        temperature=0.3,
    )
    mode = state.get("output_mode", "executive")
    system = (
        "You are a business writer for insurance. Produce a structured deliverable in JSON with exactly these keys: "
        '"executive_summary" (max ~150 words), "client_email" (short email body), '
        '"action_items" (list of objects with "owner", "task", "due_date", "confidence"). '
        "Base everything on the provided plan and research notes only. "
        f"Output mode: {mode} (executive = concise, analyst = more detail). "
        "Output valid JSON only, no markdown or preamble."
    )
    user = (
        f"Question: {state['question']}\n\nGoal: {state['goal']}\n\n"
        f"Plan: {state.get('plan', '')}\n\n"
        f"Research notes: {state.get('research_notes', '')}"
    )
    start = time.perf_counter()
    errors = 0
    response = None
    try:
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        raw = response.content if hasattr(response, "content") else str(response)
        raw = re.sub(r"^```\w*\n?", "", raw).strip()
        raw = re.sub(r"\n?```\s*$", "", raw)
        draft = json.loads(raw)
    except Exception as e:
        draft = {
            "executive_summary": f"Draft generation failed: {e}",
            "client_email": "",
            "action_items": [],
        }
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
        "draft": draft,
        "trace": [
            {
                "agent": "writer",
                "notes": "Produced structured deliverable",
                "input": {"output_mode": mode},
                "output": {
                    "draft_keys": list(draft.keys()) if isinstance(draft, dict) else [],
                    "latency_ms": latency_ms,
                    "token_usage": token_usage,
                    "errors": errors,
                },
            }
        ],
    }
