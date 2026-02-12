## Enterprise Multi-Agent Copilot (Insurance Scenario)

This project implements **Project #6: Enterprise Multi-Agent Copilot** from Giga Academy, using:

- **LangGraph + LangChain-core** for multi-agent orchestration
- **OpenAI** (GPT-4.1-mini) for reasoning and generation
- **FAISS** + LangChain for retrieval over insurance PDFs
- **Streamlit** for a minimal UI

The system turns a business request into a structured, decision-ready deliverable using a coordinated set of agents:
Planner → Researcher → Writer → Verifier → Deliver.

### Repository Structure

- `app/` – Streamlit UI (`main.py`)
- `agents/` – agent definitions (`planner`, `researcher`, `writer`, `verifier`) and `graph` (LangGraph workflow)
- `retrieval/` – `vector_store.py` (document loading + FAISS vector search)
- `data/` – sample insurance PDFs in `insurance_docs/`
- `eval/` – (optional) test prompts

### How to Run Locally

1. **Create and activate a virtual environment (recommended)**

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set your OpenAI API key and model config**

```bash
set OPENAI_API_KEY=your_key_here
set MODEL_MAIN=gpt-4.1-mini
set MODEL_EVAL=gpt-4.1-mini
```

4. **Run the Streamlit app**

```bash
streamlit run app/main.py
```

5. **Use the UI**

- Enter a **Business Question** and **Goal** (insurance scenario).
- Click **“Run Copilot”**.
- The app will:
  - Build a FAISS index over the PDFs in `data/insurance_docs/`
  - Run the LangGraph workflow:
    - Planner: decomposes the task and creates a plan
    - Researcher: retrieves grounded notes with citations
    - Writer: produces a structured draft deliverable
    - Verifier: checks for hallucinations and marks unsupported claims as `"Not found in sources."`
  - Display:
    - **Executive Summary** (max ~150 words)
    - **Client-ready Email**
    - **Action List** (owner, due date, confidence)
    - **Sources & citations** (`DocumentName | page | chunk`)
  - Show a **Trace Log** of which agent did what, with inputs/outputs.

### Nice-to-have Features Implemented

- **Prompt injection defense rules**
  - Research agent treats retrieved content as untrusted data and filters suspicious chunks.
  - Verifier includes deterministic guardrails for unsupported/unknown citations.
- **Multi-output mode (executive vs analyst)**
  - Streamlit sidebar includes output mode selector.
  - Mode is propagated through workflow and used by writer behavior.
- **Observability table (latency, tokens, errors)**
  - Per-agent and total metrics shown in UI.
  - Included in eval outputs as well.
- **Evaluation set with 10 test questions**
  - `eval/test_prompts.txt` is included.

### Evaluations

You can add your own test questions in the `eval/` folder, for example:

- `eval/test_prompts.txt` with 5–10 representative insurance business questions.

Then manually run them through the UI or script the workflow with `agents.graph.run_copilot`.

Run the eval script with your cheaper eval model:

```bash
python eval/run_eval.py
```

This reads `eval/test_prompts.txt`, runs all prompts with `MODEL_EVAL`, and writes `eval/eval_results.json`.

