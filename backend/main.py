import asyncio
import sys
import logging

from config.paths import BACKEND_LOG_FILE, ensure_storage_dirs


ensure_storage_dirs()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(BACKEND_LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI
from rag.rag_initializer import initialize_vector_store
from rag.answer_generator import generate_answer
from rag.combined_graph import build_combined_graph_runtime, run_combined_graph
from rag.hexnode_graph import build_hexnode_graph_runtime
from rag.hexnode_tools import handle_hexnode_question
from rag.keka_rag.service import initialize_keka_service
from rag.keka_rag.tools import handle_keka_question
from fastapi.middleware.cors import CORSMiddleware


from pydantic import BaseModel


class Query(BaseModel):
    question: str
    source: str = "default"

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
keka_agent = None
hexnode_graph_runtime = None
combined_graph_runtime = None
last_keka_init_error = None
last_combined_init_error = None


def init_keka_pipeline():
    global keka_agent, keka_rag_chain, keka_retriever, last_keka_init_error

    if keka_agent is not None and keka_retriever is not None and keka_rag_chain is not None:
        return

    try:
        keka_service = initialize_keka_service()
        keka_retriever = keka_service.retriever
        keka_rag_chain = keka_service.rag_chain
        keka_agent = keka_service.agent
        last_keka_init_error = None
    except Exception as exc:
        last_keka_init_error = str(exc)
        raise


def init_hexnode_graph():
    global hexnode_graph_runtime

    if hexnode_graph_runtime is not None or vector_store is None:
        return

    hexnode_graph_runtime = build_hexnode_graph_runtime(vector_store)


def init_combined_graph():
    global combined_graph_runtime, last_combined_init_error

    if combined_graph_runtime is not None:
        return

    if (
        vector_store is None
        or hexnode_graph_runtime is None
        or keka_agent is None
        or keka_retriever is None
        or keka_rag_chain is None
    ):
        return

    try:
        combined_graph_runtime = build_combined_graph_runtime(
            vector_store,
            keka_retriever,
            keka_rag_chain,
            keka_agent,
            hexnode_graph_runtime=hexnode_graph_runtime,
        )
        last_combined_init_error = None
    except Exception as exc:
        last_combined_init_error = str(exc)
        raise


def ensure_keka_ready():
    if keka_agent is not None and keka_retriever is not None and keka_rag_chain is not None:
        return None

    try:
        init_keka_pipeline()
        return None
    except Exception as exc:
        logger.exception("Keka RAG initialization failed during request: %s", exc)
        return str(exc)


def ensure_combined_ready():
    if hexnode_graph_runtime is None and vector_store is not None:
        try:
            init_hexnode_graph()
        except Exception as exc:
            logger.exception("Hexnode graph initialization failed during request: %s", exc)

    keka_error = ensure_keka_ready()
    if keka_error:
        return keka_error

    if combined_graph_runtime is None:
        try:
            init_combined_graph()
        except Exception as exc:
            logger.exception("Combined graph initialization failed during request: %s", exc)
            return str(exc)

    return None


# Run scraper once when server starts
@app.on_event("startup")
async def startup_event():
    global vector_store
    ensure_storage_dirs()
    vector_store = await initialize_vector_store()
    logger.info("Hexnode vector store ready with %s vectors", vector_store.index.ntotal)

    try:
        init_hexnode_graph()
    except ValueError as exc:
        logger.warning("Hexnode graph skipped: %s", exc)
    except Exception as exc:
        logger.exception("Hexnode graph failed to initialize: %s", exc)

    # Preload Keka RAG pipeline so source switching is ready immediately.
    try:
        init_keka_pipeline()
    except ValueError as exc:
        logger.warning("Keka RAG skipped: %s", exc)
    except Exception as exc:
        logger.exception("Keka RAG failed to initialize: %s", exc)

    try:
        init_combined_graph()
    except ValueError as exc:
        logger.warning("Combined graph skipped: %s", exc)
    except Exception as exc:
        logger.exception("Combined graph failed to initialize: %s", exc)


@app.post("/ask")
def ask(query: Query):
    logger.info("Received question for source=%s", query.source)
    question = query.question
    normalized_source = query.source.lower()

    if normalized_source == "both":
        if vector_store is None:
            return {
                "question": question,
                "chunks": [],
                "answer": "❌ Hexnode knowledge base is not initialized yet. Start the vector-store setup or enable the startup initializer before asking combined questions.",
                "source": query.source,
            }

        combined_error = ensure_combined_ready()
        if combined_error:
            return {
                "question": question,
                "chunks": [],
                "answer": f"❌ Keka RAG not available right now. Initialization failed with: {combined_error}",
                "source": query.source,
            }

        response = run_combined_graph(
            question,
            vector_store,
            keka_retriever,
            keka_rag_chain,
            keka_agent,
            runtime=combined_graph_runtime,
            hexnode_graph_runtime=hexnode_graph_runtime,
        )
    elif normalized_source in {"keka", "keka_rag"}:
        keka_error = ensure_keka_ready()
        if keka_error:
            return {
                "question": question,
                "chunks": [],
                "answer": f"❌ Keka RAG not available right now. Initialization failed with: {keka_error}",
                "source": query.source,
            }

        response = handle_keka_question(question, keka_retriever, keka_rag_chain, agent=keka_agent)
    else:
        if vector_store is None:
            return {
                "question": question,
                "chunks": [],
                "answer": "❌ Hexnode knowledge base is not initialized yet. Start the vector-store setup or enable the startup initializer before asking default-source questions.",
                "source": query.source,
            }

        response = handle_hexnode_question(
            question,
            vector_store,
            generate_answer,
            graph_runtime=hexnode_graph_runtime,
        )

    response["source"] = query.source
    return response
