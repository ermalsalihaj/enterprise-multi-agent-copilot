"""Document loading and FAISS vector search over insurance PDFs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import PROJECT_ROOT, settings


def load_pdfs(docs_dir: Path | None = None) -> list[Document]:
    """Load PDFs from docs_dir (default: data/insurance_docs) and return LangChain Documents."""
    if docs_dir is None:
        docs_dir = PROJECT_ROOT / "data" / "insurance_docs"
    if not docs_dir.exists():
        return []
    documents: list[Document] = []
    for path in sorted(docs_dir.glob("*.pdf")):
        try:
            reader = PdfReader(str(path))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": path.name, "page": i + 1},
                        )
                    )
        except Exception:
            continue
    return documents


def build_vector_store(docs_dir: Path | None = None) -> Optional[FAISS]:
    """Build FAISS index from PDFs. Requires OPENAI_API_KEY for embeddings."""
    documents = load_pdfs(docs_dir)
    if not documents:
        return None
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
    )
    splits = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    return FAISS.from_documents(splits, embeddings)


def search_sources(
    vector_store: Optional[FAISS],
    query: str,
    k: int = 8,
) -> list[dict[str, Any]]:
    """Return list of source dicts with citation and note from FAISS similarity search."""
    if vector_store is None:
        return []
    docs = vector_store.similarity_search(query, k=k)
    sources = []
    for i, doc in enumerate(docs):
        meta = doc.metadata or {}
        name = meta.get("source", "Unknown")
        page = meta.get("page", "?")
        citation = f"{name} | page {page} | chunk {i + 1}"
        sources.append({"citation": citation, "note": doc.page_content[:500]})
    return sources
