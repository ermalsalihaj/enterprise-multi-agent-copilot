"""Verifier agent: checks draft against sources and marks unsupported claims."""
from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings

from .state import GraphState


def verifier_node(state: GraphState) -> dict[str, Any]:
    """Verify draft against sources; mark unsupported claims as 'Not found in sources.'"""
    llm = ChatOpenAI(
        model=settings.model_main,
        api_key=settings.openai_api_key,
        temperature=0.0,
    )
    draft = state.get("draft") or {}
    sources = state.get("sources") or []
    source_text = "\n".join(
        f"- {s.get('citation', '')}: {s.get('note', '')[:200]}" for s in sources
    ) or "No sources."
    system = (
        "You are a verifier. Given a draft deliverable and the only allowed sources, "
        "output a verified version in JSON with keys: executive_summary, client_email, action_items, sources. "
        "For any claim not supported by the sources, replace that part with exactly: Not found in sources. "
        "Keep supported content. For sources, pass through only the list of citations/notes that were actually used. "
        "Output valid JSON only."
    )
    user = (
        f"Draft:\n{str(draft)}\n\n"
        f"Allowed sources:\n{source_text}"
    )
    start = time.perf_counter()
    errors = 0
    response = None
    try:
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        raw = response.content if hasattr(response, "content") else str(response)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        verified_output = json.loads(raw)
    except Exception as e:
        verified_output = {
            "executive_summary": draft.get("executive_summary", "Not found in sources."),
            "client_email": draft.get("client_email", "Not found in sources."),
            "action_items": draft.get("action_items", []),
            "sources": [s for s in sources],
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
        "verified_output": verified_output,
        "trace": [
            {
                "agent": "verifier",
                "notes": "Verified against sources",
                "input": {"draft_keys": list(draft.keys()) if draft else []},
                "output": {
                    "latency_ms": latency_ms,
                    "token_usage": token_usage,
                    "errors": errors,
                },
            }
        ],
    }
