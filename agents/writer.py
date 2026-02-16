"""Writer agent: produces structured deliverable (executive summary, email, action list)."""
from __future__ import annotations

import json
import re
import time
from typing import Any

from config import settings

from .llm import invoke_openai_chat
from .state import GraphState, PROMPT_INJECTION_DEFENSE


def writer_node(state: GraphState) -> dict[str, Any]:
    """Produce draft deliverable from plan and research notes."""
    mode = state.get("output_mode", "executive")
    signer = (state.get("email_signer") or "").strip() or "The Advisory Team"
    system = (
        PROMPT_INJECTION_DEFENSE + " "
        "You are a business writer for insurance. Produce a structured deliverable in JSON with exactly these keys: "
        '"executive_summary" (max ~150 words), "client_email" (short email body), '
        '"action_items" (list of objects with "owner", "task", "due_date", "confidence"). '
        "Base everything on the provided plan and research notes only. "
        "Keep the deliverable strictly professional: no jokes, no humor, no off-topic or casual content. "
        f"Output mode: {mode} (executive = concise, analyst = more detail). "
        f'End the client_email with a sign-off. Use exactly this in place of [Your Name]: "{signer}". '
        "Output valid JSON only, no markdown or preamble."
    )
    user = (
        f"Question: {state['question']}\n\nGoal: {state['goal']}\n\n"
        f"Plan: {state.get('plan', '')}\n\n"
        f"Research notes: {state.get('research_notes', '')}"
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
            temperature=0.3,
        )
        raw = (raw or "").strip()
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
        draft = json.loads(raw) if raw else {}
    except Exception as e:
        draft = {
            "executive_summary": f"Draft generation failed: {e}",
            "client_email": "",
            "action_items": [],
        }
        errors = 1
    latency_ms = int((time.perf_counter() - start) * 1000)
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
