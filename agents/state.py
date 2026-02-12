"""Shared state for the multi-agent LangGraph workflow."""
from __future__ import annotations

from operator import add
from typing import Annotated, Any, Literal, TypedDict

from typing_extensions import NotRequired


class GraphState(TypedDict):
    """State passed between planner, researcher, writer, and verifier."""

    question: str
    goal: str
    output_mode: Literal["executive", "analyst"]
    plan: NotRequired[str]
    research_notes: NotRequired[str]
    sources: NotRequired[list[dict[str, Any]]]
    draft: NotRequired[dict[str, Any]]
    verified_output: NotRequired[dict[str, Any]]
    trace: Annotated[list[dict[str, Any]], add]
