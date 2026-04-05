import os
import shutil
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rag.keka_rag.embeddings import get_embeddings


def load_vectorstore(path):
    if not os.path.exists(path):
        return None

    try:
        db = FAISS.load_local(
            path,
            get_embeddings(),
            allow_dangerous_deserialization=True
        )
        print("✅ Loaded FAISS index")
        return db
    except Exception as e:
        print(f"⚠️ Failed loading FAISS: {e}")
        return None


def create_vectorstore(chunks, path):
    if not chunks:
        raise ValueError("No chunks provided")

    print("🚀 Creating FAISS index...")

    db = FAISS.from_documents(
        documents=chunks,
        embedding=get_embeddings()
    )

    os.makedirs(path, exist_ok=True)
    db.save_local(path)

    print("✅ FAISS saved")
    return db


def get_vectorstore(chunks=None, path="keka_faiss"):
    db = load_vectorstore(path)

    if db:
        return db

    if chunks is None:
        raise ValueError("Chunks required to create FAISS")

    return create_vectorstore(chunks, path)

