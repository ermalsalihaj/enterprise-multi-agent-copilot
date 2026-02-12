from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on path so "agents" and "retrieval" can be imported
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
load_dotenv(_project_root / ".env")

import streamlit as st

from agents.graph import run_copilot
from config import settings


def _render_action_items(action_items):
    if not action_items:
        st.write("No action items returned.")
        return
    for idx, item in enumerate(action_items, start=1):
        owner = item.get("owner", "Owner")
        task = item.get("task", "")
        due = item.get("due_date", "N/A")
        conf = item.get("confidence", "N/A")
        st.markdown(
            f"**{idx}. {owner}** – {task}  \n"
            f"*Due*: {due} | *Confidence*: {conf}"
        )


def _render_sources(sources):
    if not sources:
        st.write("No sources returned.")
        return
    for s in sources:
        if isinstance(s, dict):
            citation = s.get("citation", "Unknown citation")
            note = s.get("note", "")
        else:
            citation = str(s)
            note = ""
        st.markdown(f"- **{citation}** – {note}")


def main():
    st.set_page_config(page_title="Enterprise Multi-Agent Copilot", layout="wide")
    st.title("Enterprise Multi-Agent Copilot – Insurance Scenario")

    st.sidebar.header("How To Use")
    st.sidebar.write(
        "1) Enter your business question.\n"
        "2) Add a clear goal for the deliverable.\n"
        "3) Click **Run Copilot** to generate a verified output with citations."
    )
    st.sidebar.caption(
        "Models: regular runs use MODEL_MAIN, eval runs use MODEL_EVAL."
    )
    output_mode = st.sidebar.selectbox(
        "Output Mode",
        options=["executive", "analyst"],
        index=0,
        help="Executive is concise, Analyst is more detailed.",
    )

    question = st.text_input("Business Question", placeholder="e.g., How can we reduce motor insurance claim leakage in EMEA?")
    goal = st.text_area(
        "Goal",
        placeholder="e.g., Provide 3–5 actionable initiatives with pros/cons and implementation considerations.",
    )

    if st.button("Run Copilot", type="primary") and question and goal:
        if not settings.openai_api_key:
            st.error("OPENAI_API_KEY is not set in environment.")
            return

        with st.spinner("Running multi-agent workflow..."):
            result = run_copilot(question=question, goal=goal, output_mode=output_mode)

        col_main, col_trace = st.columns([2, 1])

        with col_main:
            st.subheader("Final Deliverable (Verified)")
            verified = result.get("verified_output", {}) or {}

            exec_summary = verified.get("executive_summary") or "Not found in sources."
            client_email = verified.get("client_email") or "Not found in sources."
            action_items = verified.get("action_items", [])
            sources = verified.get("sources", [])

            st.markdown("### Executive Summary")
            st.write(exec_summary)

            st.markdown("### Client-ready Email")
            st.write(client_email)

            st.markdown("### Action List")
            _render_action_items(action_items)

            st.markdown("### Sources & Citations")
            _render_sources(sources)

            observability = result.get("observability", {})
            per_agent = observability.get("per_agent", [])
            totals = observability.get("totals", {})
            st.markdown("### Observability")
            if per_agent:
                st.table(per_agent)
            st.caption(
                "Totals - latency_ms: "
                f"{totals.get('latency_ms', 0)}, prompt_tokens: {totals.get('prompt_tokens', 0)}, "
                f"completion_tokens: {totals.get('completion_tokens', 0)}, "
                f"total_tokens: {totals.get('total_tokens', 0)}, errors: {totals.get('errors', 0)}"
            )

        with col_trace:
            st.subheader("Trace Log")
            trace = result.get("trace", [])
            if not trace:
                st.write("No trace events.")
            else:
                for event in trace:
                    st.markdown(f"**Agent**: {event.get('agent')}")
                    st.caption(event.get("notes", ""))
                    with st.expander("Input", expanded=False):
                        st.code(json.dumps(event.get("input", {}), indent=2))
                    with st.expander("Output", expanded=False):
                        st.code(json.dumps(event.get("output", {}), indent=2))


if __name__ == "__main__":
    main()

