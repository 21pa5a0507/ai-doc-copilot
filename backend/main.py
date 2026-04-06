import asyncio
import sys
import logging
import pickle
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI
from rag.answer_generator import generate_answer, rewrite_query
from rag.embendings import get_embending as embed_text
from rag.vector_store import VectorStore
from rag.scraper import scrap_website
from rag.rag_initializer import initialize_vector_store
from rag.keka_rag.loaders import load_pdfs
from rag.keka_rag.splitter import split_documents
from rag.keka_rag.vector_store import get_vectorstore as get_keka_vectorstore
from rag.keka_rag.retriever import get_retriever as get_keka_retriever
from rag.keka_rag.rag_chain import get_rag_chain as get_keka_rag_chain
from fastapi.middleware.cors import CORSMiddleware


from pydantic import BaseModel

class Query(BaseModel):
    question: str
    source: str = "default"

CACHE_FILE = "rag/keka_faiss/keka_docs_cache.pkl"

def load_cached_docs():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load cached docs: {e}")
    return None

def save_cached_docs(docs):
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(docs, f)
        print("✅ Cached docs saved to file")
    except Exception as e:
        print(f"⚠️ Failed to save cached docs: {e}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


vector_store = None
keka_rag_chain = None
keka_retriever = None
keka_docs = None


def init_keka_pipeline():
    global keka_rag_chain, keka_retriever, keka_docs

    if keka_rag_chain is not None and keka_retriever is not None:
        return

    # Load cached docs first
    keka_docs = load_cached_docs()

    try:
        vectorstore = get_keka_vectorstore()
        print("✅ Loaded existing Keka FAISS index")
    except Exception as e:
        print(f"⚠️ Keka FAISS load failed: {e}. Building new index...")
        if keka_docs is None:
            docs = load_pdfs()
            keka_docs = split_documents(docs)
            save_cached_docs(keka_docs)
        vectorstore = get_keka_vectorstore(keka_docs)

    # Load docs if not cached
    if keka_docs is None:
        docs = load_pdfs()
        keka_docs = split_documents(docs)
        save_cached_docs(keka_docs)

    keka_retriever = get_keka_retriever(vectorstore, keka_docs)
    keka_rag_chain = get_keka_rag_chain(keka_retriever)
    print("✅ Keka RAG pipeline ready")


# Run scraper once when server starts
@app.on_event("startup")
async def startup_event():
    global vector_store
    vector_store = await initialize_vector_store()
    print("Vectors stored:", vector_store.index.ntotal)

    # Preload Keka RAG pipeline so source switching is ready immediately.
    try:
        init_keka_pipeline()
    except ValueError as e:
        print(f"⚠️ Keka RAG skipped: {e}")
    except Exception as e:
        print(f"⚠️ Keka RAG failed to initialize: {e}")


@app.post("/ask")
def ask(query: Query):
    print(f"Received question: {query.question} (source={query.source})")
    question = query.question

    if query.source.lower() in {"keka", "keka_rag"}:
        if keka_retriever is None or keka_rag_chain is None:
            return {
                "question": question,
                "chunks": [],
                "answer": "❌ Keka RAG not available. Please set GOOGLE_API_KEY environment variable.",
                "source": query.source
            }

        docs = keka_retriever.invoke(question)
        print(f"🔍 Keka retrieved {len(docs)} docs for question: {question}")

        chunks = [
            {
                "title": doc.metadata.get("file_name", "Keka document"),
                "content": doc.page_content,
                "source": doc.metadata.get("source", "keka"),
            }
            for doc in docs
        ]

        answer = keka_rag_chain(question, debug=False)
    else:
        query_embedding = embed_text(question)

        chunks = vector_store.search(query_embedding, question)
        print(f"🔍 Retrieved {len(chunks)} chunks for question: {question}")

        answer = generate_answer(question, chunks)
    return {
        "question": question,
        "chunks": chunks,
        "answer": answer,
        "source": query.source
    }