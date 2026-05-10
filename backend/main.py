import asyncio
import sys
import logging
import os
import time

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

from fastapi import FastAPI, Header, HTTPException
from rag.rag_initializer import initialize_vector_store
from rag.answer_generator import generate_answer, get_greeting_response
from rag.combined_graph import build_combined_graph_runtime, run_combined_graph
from rag.hexnode_graph import build_hexnode_graph_runtime
from rag.hexnode_tools import handle_hexnode_question
from rag.keka_rag.service import initialize_keka_service
from rag.keka_rag.tools import handle_keka_question
from rag.workflow_agent.jira_client import process_delete_approvals
from rag.workflow_agent.main import run_workflow_agent
from fastapi.middleware.cors import CORSMiddleware


from pydantic import BaseModel


class ChatHistoryMessage(BaseModel):
    role: str
    text: str


class Query(BaseModel):
    question: str
    source: str = "default"
    history: list[ChatHistoryMessage] = []

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

    if keka_retriever is not None and keka_rag_chain is not None:
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
        or keka_retriever is None
    ):
        return

    try:
        combined_graph_runtime = build_combined_graph_runtime(
            vector_store,
            keka_retriever,
        )
        last_combined_init_error = None
    except Exception as exc:
        last_combined_init_error = str(exc)
        raise


def ensure_keka_ready():
    if keka_retriever is not None and keka_rag_chain is not None:
        return None

    try:
        init_keka_pipeline()
        return None
    except Exception as exc:
        logger.exception("Keka RAG initialization failed during request: %s", exc)
        return str(exc)


def ensure_combined_ready():
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

    # Keep Keka ready for source switching after startup.
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


@app.get("/health")
def health_check():
    hexnode_ready = vector_store is not None
    keka_ready = keka_retriever is not None and keka_rag_chain is not None
    combined_ready = combined_graph_runtime is not None

    return {
        "status": "ok" if hexnode_ready and keka_ready else "degraded",
        "hexnode_ready": hexnode_ready,
        "keka_ready": keka_ready,
        "combined_ready": combined_ready,
    }


@app.post("/workflow/process-delete-approvals")
def process_workflow_delete_approvals(
    x_workflow_scheduler_token: str = Header(default=""),
):
    expected_token = os.getenv("WORKFLOW_SCHEDULER_TOKEN", "")

    if expected_token and x_workflow_scheduler_token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid scheduler token")

    return process_delete_approvals()


@app.post("/ask")
def ask(query: Query):
    request_start = time.perf_counter()
    logger.info("Received question for source=%s", query.source)
    question = query.question
    normalized_source = query.source.lower()
    greeting_response = get_greeting_response(question, query.source)

    if greeting_response is not None:
        greeting_response["timings"] = {
            "request_total_seconds": round(time.perf_counter() - request_start, 4),
        }
        return greeting_response

    if normalized_source == "both":
        if vector_store is None:
            return {
                "question": question,
                "chunks": [],
                "answer": "❌ Hexnode knowledge base is not initialized yet. Start the vector-store setup or enable the startup initializer before asking combined questions.",
                "source": query.source,
                "timings": {
                    "request_total_seconds": round(time.perf_counter() - request_start, 4),
                },
            }

        combined_error = ensure_combined_ready()
        if combined_error:
            return {
                "question": question,
                "chunks": [],
                "answer": f"❌ Keka RAG not available right now. Initialization failed with: {combined_error}",
                "source": query.source,
                "timings": {
                    "request_total_seconds": round(time.perf_counter() - request_start, 4),
                },
            }

        response = run_combined_graph(
            question,
            vector_store,
            keka_retriever,
            runtime=combined_graph_runtime,
        )
    elif normalized_source in {"workflow", "workflow_agent", "support"}:
        try:
            response = run_workflow_agent(
                question,
                vector_store=vector_store,
                history=[message.model_dump() for message in query.history],
            )
        except Exception as exc:
            logger.exception("Workflow agent failed during request: %s", exc)
            return {
                "question": question,
                "chunks": [],
                "answer": "❌ Workflow agent is temporarily unavailable. Please try again shortly.",
                "source": query.source,
                "tool_calls": [],
                "timings": {
                    "request_total_seconds": round(time.perf_counter() - request_start, 4),
                },
            }
    elif normalized_source in {"keka", "keka_rag"}:
        keka_error = ensure_keka_ready()
        if keka_error:
            return {
                "question": question,
                "chunks": [],
                "answer": f"❌ Keka RAG not available right now. Initialization failed with: {keka_error}",
                "source": query.source,
                "timings": {
                    "request_total_seconds": round(time.perf_counter() - request_start, 4),
                },
            }

        response = handle_keka_question(question, keka_retriever, keka_rag_chain, agent=keka_agent)
    else:
        if vector_store is None:
            return {
                "question": question,
                "chunks": [],
                "answer": "❌ Hexnode knowledge base is not initialized yet. Start the vector-store setup or enable the startup initializer before asking default-source questions.",
                "source": query.source,
                "timings": {
                    "request_total_seconds": round(time.perf_counter() - request_start, 4),
                },
            }

        response = handle_hexnode_question(
            question,
            vector_store,
            generate_answer,
            graph_runtime=hexnode_graph_runtime,
        )

    response["source"] = query.source
    timings = dict(response.get("timings", {}))
    timings["request_total_seconds"] = round(time.perf_counter() - request_start, 4)
    response["timings"] = timings
    return response
