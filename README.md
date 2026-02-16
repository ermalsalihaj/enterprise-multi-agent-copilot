# Enterprise Multi-Agent Copilot (Insurance Scenario)

This repository implements **Project #6: Enterprise Multi-Agent Copilot**. It is a multi-agent system that turns a business question and optional goal into a structured, decision-ready deliverable (executive summary, client email, action list) grounded in retrieved documents. The scenario is **insurance**: documents in `data/insurance_docs/` are PDFs that the system searches and cites.

The stack is **LangGraph** for orchestration, **OpenAI** (e.g. GPT-4.1-mini) for planning and generation, **FAISS** (via LangChain) for vector search over the PDFs, and **Streamlit** for the UI.

---

## What the system does

A user enters a **business question** (e.g. “What are the main drivers of insurance growth in our key markets?”) and optionally a **goal** (e.g. “Produce an analyst-mode deliverable with executive summary and 90-day action list”). The system then:

1. **Plans** — The Planner agent decomposes the task into a step-by-step plan.
2. **Researches** — The Researcher agent queries the document index (FAISS), retrieves relevant chunks, and produces research notes with citations.
3. **Drafts** — The Writer agent turns the plan and research notes into a structured deliverable: executive summary (max ~150 words), client-ready email, and action list (owner, task, due date, confidence).
4. **Verifies** — The Verifier agent checks every claim in the draft against the retrieved sources. Any claim not supported by the sources is replaced with the exact phrase **“Not found in sources.”** so the final output is citation-grounded.

The result is shown in the UI as a **Final deliverable (verified)** with expandable sections, plus **Sources and citations** and a **Trace log** of which agent did what (with inputs and outputs). An **Observability** section shows per-agent and total latency, token counts, and errors.

---

## Repository structure

- **`app/`** — Streamlit application. `main.py` is the entry point: it provides the form (business question, optional goal, output mode, optional email sign-off), calls the LangGraph workflow when the user clicks Run Copilot, and displays the verified deliverable, sources, trace log, and observability table.

- **`agents/`** — Agent definitions and the workflow graph. The **Planner** (`planner.py`), **Researcher** (`researcher.py`), **Writer** (`writer.py`), and **Verifier** (`verifier.py`) are LangGraph nodes. The **graph** (`graph.py`) wires them in sequence (Plan → Research → Write → Verify → End) and exposes `run_copilot()`. The **state** (`state.py`) defines the shared state (question, goal, plan, research_notes, sources, draft, verified_output, trace) and shared prompt-injection defense text. The **LLM** (`llm.py`) is a thin wrapper around the OpenAI API used by all agents so token usage is read reliably from the response.

- **`retrieval/`** — Document loading and vector search. `vector_store.py` loads PDFs from `data/insurance_docs/`, splits them into chunks, builds a FAISS index with OpenAI embeddings, and exposes `search_sources()` so the Researcher can retrieve cited excerpts.

- **`data/`** — Root for input documents. PDFs live in `data/insurance_docs/` and are indexed when the app or eval runs. See `data/README.md` for what this folder contains and how citations are formatted.

- **`eval/`** — Evaluation prompts and script. `test_prompts.txt` holds one question per line (optional `||goal` suffix). `run_eval.py` reads that file, runs each line through `run_copilot()` using the eval model, and writes `eval_results.json` with outputs and observability.

- **`config.py`** (project root) — Loads environment variables (e.g. `OPENAI_API_KEY`, `MODEL_MAIN`, `MODEL_EVAL`, `EMBEDDING_MODEL`) and exposes a `settings` object used by the app and agents.

---

## How to run the application

Running the app locally means starting the Streamlit server and opening the UI in the browser. The following explains what is required and what happens.

**Environment**

- **Python** — A supported Python version (e.g. 3.10+) is required. Using a virtual environment is recommended so dependencies do not conflict with other projects. Creating and activating one looks like:
  - `python -m venv .venv`
  - On Windows: `.venv\Scripts\activate`  
  - On macOS/Linux: `source .venv/bin/activate`

- **Dependencies** — From the project root, `pip install -r requirements.txt` installs LangGraph, LangChain, OpenAI, FAISS, Streamlit, pypdf, and the other packages the app and agents use.

- **OpenAI API key** — The copilot calls the OpenAI API for both chat (planning, research, writing, verification) and embeddings (vector index). The key is read from the environment. Typically it is set in a shell or in a `.env` file in the project root (e.g. `OPENAI_API_KEY=sk-...`). Optional env vars include `MODEL_MAIN` (defaults to a model name such as `gpt-4.1-mini`), `MODEL_EVAL` (used by the eval script), and `EMBEDDING_MODEL` for the retrieval layer.

**Starting the UI**

From the project root, run:

```bash
streamlit run app/main.py
```

Streamlit starts a local server and prints a URL (often `http://localhost:8501`). Opening that URL in a browser shows the Enterprise Multi-Agent Copilot UI: a business question field, an optional goal field, sidebar options (ready-made questions, output mode, email sign-off), and a Run Copilot button. When the user clicks Run Copilot, the app builds the FAISS index from the PDFs in `data/insurance_docs/` (if not already in memory), invokes the LangGraph workflow, and then displays the Final deliverable (verified), Sources and citations, Trace log, and Observability table. The project is designed to run locally within a few minutes (install, set key, run the command above).

---

## What you see in the UI

- **Final deliverable (verified)** — Expandable sections for **Executive summary**, **Client-ready email**, and **Action list** (owner, task, due date, confidence). All of this comes from the Verifier’s output and is intended to be grounded in the retrieved sources; unsupported claims are replaced with “Not found in sources.”

- **Sources and citations** — A list of sources that were used, in the form `DocumentName | page X | chunk Y`. No excerpt text is shown here; the format is explained in `data/README.md`.

- **Observability** — A table of per-agent metrics (latency_ms, prompt_tokens, completion_tokens, total_tokens, errors) and a Totals row. This helps with monitoring cost and performance.

- **Trace log** — A step-by-step view of the workflow (Planner, Researcher, Writer, Verifier). Each step can be expanded to show Input and Output as plain text (no JSON formatting), so it is clear which agent did what.

---

## Nice-to-have features included

- **Prompt injection defense** — The app checks the business question against a list of known injection phrases (e.g. “disregard prior rules”, “ignore all previous instructions”). If a match is found, the run is blocked and an error message is shown. In addition, every agent’s system prompt includes a shared instruction to ignore override attempts, jokes, and non-business content and to output only professional, source-grounded material. The Verifier is also instructed to remove or replace jokes and unsupported content with “Not found in sources.”

- **Multi-output mode (executive vs analyst)** — The sidebar has an output mode selector. The chosen mode is passed through the workflow and influences how concise or detailed the Writer’s deliverable is.

- **Observability table** — Per-agent and total latency, token counts, and errors are displayed in the UI and are also available in the object returned by `run_copilot()` and in the eval script output.

- **Evaluation set** — The `eval/` folder contains `test_prompts.txt` with multiple test questions (e.g. 10). Each line can optionally include a goal after `||`. This supports batch evaluation and regression checks.

---

## Evaluations

The **eval** folder is used for batch runs and testing. `test_prompts.txt` holds one prompt per line. A line can be a question only, or `question||goal`. Lines starting with `#` are skipped. The script `run_eval.py` reads this file, invokes `run_copilot()` for each line (using the eval model from config), and writes results to `eval_results.json`. Running the eval script is done from the project root with:

```bash
python eval/run_eval.py
```

The script expects `eval/test_prompts.txt` to exist. The JSON output contains the verified outputs and observability for each prompt, which can be used to compare runs or to validate that the system meets requirements (citations, “Not found in sources.” for unsupported claims, trace visibility, etc.).

---

## Project requirements alignment

The implementation matches Project #6’s requirements: multi-agent workflow (Plan → Research → Draft → Verify → Deliver), four agents (Planner, Researcher, Writer, Verifier), retrieval over 5–15 documents in `data/insurance_docs/`, citations in the form DocumentName + page + chunk, verifier replacing unsupported claims with “Not found in sources.”, structured deliverable (executive summary, client email, action list with owner/due date/confidence), trace log visible in the UI, and the option to run locally within minutes. The chosen industry is insurance and the chosen stack is LangGraph. The listed nice-to-haves (prompt injection defense, multi-output mode, observability table, evaluation set with 10 test questions) are all present.
