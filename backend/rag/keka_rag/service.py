import pickle
from dataclasses import dataclass
from typing import Any, List

from config.paths import KEKA_DOCS_CACHE, KEKA_FAISS_DIR, ensure_storage_dirs
from rag.keka_rag.keka_agent import build_keka_agent
from rag.keka_rag.loaders import load_pdfs
from rag.keka_rag.rag_chain import get_rag_chain
from rag.keka_rag.retriever import get_retriever
from rag.keka_rag.splitter import split_documents
from rag.keka_rag.vector_store import get_vectorstore


@dataclass
class KekaService:
    docs: List[Any]
    vectorstore: Any
    retriever: Any
    rag_chain: Any
    agent: Any


def load_cached_docs():
    if KEKA_DOCS_CACHE.exists():
        try:
            with KEKA_DOCS_CACHE.open("rb") as file:
                return pickle.load(file)
        except Exception as exc:
            print(f"Failed to load cached docs: {exc}")
    return None


def save_cached_docs(docs):
    try:
        ensure_storage_dirs()
        with KEKA_DOCS_CACHE.open("wb") as file:
            pickle.dump(docs, file)
        print("Cached docs saved to file")
    except Exception as exc:
        print(f"Failed to save cached docs: {exc}")


def _load_or_create_docs():
    docs = load_cached_docs()

    if docs is None:
        docs = split_documents(load_pdfs())
        save_cached_docs(docs)

    return docs


def initialize_keka_service():
    ensure_storage_dirs()
    docs = load_cached_docs()

    try:
        vectorstore = get_vectorstore(path=KEKA_FAISS_DIR)
        print("Loaded existing Keka FAISS index")
    except Exception as exc:
        print(f"Keka FAISS load failed: {exc}. Building new index...")
        docs = docs or _load_or_create_docs()
        vectorstore = get_vectorstore(docs, path=KEKA_FAISS_DIR)

    docs = docs or _load_or_create_docs()
    retriever = get_retriever(vectorstore, docs)
    rag_chain = get_rag_chain(retriever)
    agent = build_keka_agent(retriever)

    print("Keka RAG pipeline ready")
    return KekaService(
        docs=docs,
        vectorstore=vectorstore,
        retriever=retriever,
        rag_chain=rag_chain,
        agent=agent,
    )
