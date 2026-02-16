# Data Folder

This folder holds the documents used for retrieval by the multi-agent copilot. All content is **public or synthetic**—no confidential or client data.

## What this folder is for

The **data** folder is the root for all input documents that the copilot can search and cite. The retrieval layer reads from a subfolder here (`insurance_docs/`), loads PDFs, and builds a vector index so the Research agent can fetch relevant excerpts for the Planner and Writer.

## Contents

- **`insurance_docs/`** — PDF files that the retrieval layer indexes. The system looks for `*.pdf` files in this directory. Each PDF is read with pypdf (text per page), then split into chunks (800 characters, 150 overlap) and embedded for FAISS similarity search. The index is built when the copilot or eval runs; it reflects whatever PDFs are present in `insurance_docs/` at that time.

## Citation format

Sources appear in the UI and in the final deliverable in this form:

- **`DocumentName | page X | chunk Y`**

For example: `global_insurance_growth_report.pdf | page 47 | chunk 1`

- **DocumentName** is the PDF filename.
- **page** is the 1-based page number in that document.
- **chunk** is the index of the text chunk in the retrieved set (for reference).

## Data policy

- All content in this folder is **public or synthetic**. No confidential Genpact or client data is included.
- Documents are used **only** for grounded answer generation and citations.
- Document content is not stored outside the FAISS index that is built at runtime (or when the eval script runs).
