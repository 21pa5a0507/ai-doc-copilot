import logging
import pickle
from dataclasses import dataclass
from typing import Any, List

from config.paths import (
    KEKA_CHUNK_CACHE,
    KEKA_FAISS_EMB_DIR,
    LEGACY_KEKA_DOCS_CACHE,
    LEGACY_KEKA_FAISS_DIR,
    ensure_storage_dirs,
)
from rag.keka_rag.keka_agent import build_keka_agent
from rag.keka_rag.loaders import load_pdfs
from rag.keka_rag.rag_chain import get_rag_chain
from rag.keka_rag.retriever import get_retriever
from rag.keka_rag.splitter import split_documents
from rag.keka_rag.vector_store import get_vectorstore


logger = logging.getLogger(__name__)


@dataclass
class KekaService:
    docs: List[Any]
    vectorstore: Any
    retriever: Any
    rag_chain: Any
    agent: Any


def load_cached_docs():
    for cache_path in (KEKA_CHUNK_CACHE, LEGACY_KEKA_DOCS_CACHE):
        if not cache_path.exists():
            continue
        try:
            with cache_path.open("rb") as file:
                return pickle.load(file)
        except Exception as exc:
            logger.warning("Failed to load cached Keka docs: %s", exc)
    return None


def save_cached_docs(docs):
    try:
        ensure_storage_dirs()
        with KEKA_CHUNK_CACHE.open("wb") as file:
            pickle.dump(docs, file)
        logger.info("Saved cached Keka docs")
    except Exception as exc:
        logger.warning("Failed to save cached Keka docs: %s", exc)


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
        vectorstore = get_vectorstore(path=KEKA_FAISS_EMB_DIR, legacy_path=LEGACY_KEKA_FAISS_DIR)
        logger.info("Loaded existing Keka FAISS index")
    except Exception as exc:
        logger.warning("Keka FAISS load failed: %s. Building a new index.", exc)
        docs = docs or _load_or_create_docs()
        vectorstore = get_vectorstore(docs, path=KEKA_FAISS_EMB_DIR, legacy_path=LEGACY_KEKA_FAISS_DIR)

    docs = docs or _load_or_create_docs()
    retriever = get_retriever(vectorstore, docs)
    rag_chain = get_rag_chain(retriever)
    agent = build_keka_agent(retriever)

    logger.info("Keka RAG pipeline ready")
    return KekaService(
        docs=docs,
        vectorstore=vectorstore,
        retriever=retriever,
        rag_chain=rag_chain,
        agent=agent,
    )
