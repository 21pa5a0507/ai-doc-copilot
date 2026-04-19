import os
import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from config.paths import KEKA_FAISS_DIR
from rag.keka_rag.embeddings import get_embeddings


logger = logging.getLogger(__name__)


def load_vectorstore(path):
    path = Path(path)

    if not path.exists():
        return None

    try:
        db = FAISS.load_local(
            str(path),
            get_embeddings(),
            allow_dangerous_deserialization=True
        )
        logger.info("Loaded Keka FAISS index")
        return db
    except Exception as exc:
        logger.warning("Failed loading Keka FAISS index: %s", exc)
        return None


def create_vectorstore(chunks, path):
    if not chunks:
        raise ValueError("No chunks provided")

    path = Path(path)
    logger.info("Creating Keka FAISS index")

    db = FAISS.from_documents(
        documents=chunks,
        embedding=get_embeddings()
    )

    os.makedirs(path, exist_ok=True)
    db.save_local(str(path))

    logger.info("Saved Keka FAISS index")
    return db


def get_vectorstore(chunks=None, path=KEKA_FAISS_DIR):
    db = load_vectorstore(path)

    if db:
        return db

    if chunks is None:
        raise ValueError("Chunks required to create FAISS")

    return create_vectorstore(chunks, path)

