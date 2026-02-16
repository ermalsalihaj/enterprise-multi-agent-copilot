"""Verifier agent: checks draft against sources and marks unsupported claims."""
from __future__ import annotations

import json
import time
from typing import Any

from config import settings

from .llm import invoke_openai_chat
from .state import GraphState, PROMPT_INJECTION_DEFENSE


def verifier_node(state: GraphState) -> dict[str, Any]:
    """Verify draft against sources; mark unsupported claims as 'Not found in sources.'"""
    draft = state.get("draft") or {}
    sources = state.get("sources") or []
    source_text = "\n".join(
        f"- {s.get('citation', '')}: {s.get('note', '')[:200]}" for s in sources
    ) or "No sources."
    system = (
        PROMPT_INJECTION_DEFENSE + " "
        "You are a verifier. Given a draft deliverable and the only allowed sources, "
        "output a verified version in JSON with keys: executive_summary, client_email, action_items, sources. "
        "For any claim not supported by the sources, replace that part with exactly: Not found in sources. "
        "Remove or replace with 'Not found in sources.' any jokes, humor, or non-business content in the draft. "
        "Keep supported content. For sources, pass through only the list of citations/notes that were actually used. "
        "Output valid JSON only."
    )
    user = (
        f"Draft:\n{str(draft)}\n\n"
        f"Allowed sources:\n{source_text}"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    start = time.perf_counter()
    errors = 0
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        raw, token_usage = invoke_openai_chat(
            settings.model_main,
            settings.openai_api_key,
            messages,
            temperature=0.0,
        )
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        verified_output = json.loads(raw) if raw else {}
    except Exception:
        verified_output = {
            "executive_summary": draft.get("executive_summary", "Not found in sources."),
            "client_email": draft.get("client_email", "Not found in sources."),
            "action_items": draft.get("action_items", []),
            "sources": [s for s in sources],
        }
        errors = 1

    # Ensure each source has both citation and note (chunk preview) for the UI
    citation_to_source = {s.get("citation", ""): s for s in sources}
    normalized = []
    for item in verified_output.get("sources", []):
        if isinstance(item, dict):
            cit = item.get("citation", "")
            full = citation_to_source.get(cit, item)
            normalized.append({"citation": cit or full.get("citation", "?"), "note": full.get("note", item.get("note", ""))})
        else:
            cit = str(item).strip()
            full = citation_to_source.get(cit, {})
            normalized.append({"citation": cit or "?", "note": full.get("note", "")})
    verified_output["sources"] = normalized
    latency_ms = int((time.perf_counter() - start) * 1000)
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
