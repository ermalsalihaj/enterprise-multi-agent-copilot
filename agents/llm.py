"""Direct OpenAI chat call so we always get token usage from the API response."""
from __future__ import annotations

from openai import OpenAI


def invoke_openai_chat(
    model: str,
    api_key: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
) -> tuple[str, dict[str, int]]:
    """Call OpenAI chat completions and return (content, token_usage)."""
    usage_out = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not api_key:
        return "", usage_out
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    content = ""
    if response.choices:
        msg = response.choices[0].message
        if hasattr(msg, "content") and msg.content:
            content = msg.content
    if getattr(response, "usage", None):
        u = response.usage
        usage_out = {
            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
            "total_tokens": getattr(u, "total_tokens", 0) or 0,
        }
    return content, usage_out
