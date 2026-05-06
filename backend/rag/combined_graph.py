from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.prompts import ChatPromptTemplate
import logging

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - handled by caller if langgraph is unavailable
    END = START = StateGraph = None

from rag.content import content_to_text
from rag.gemini_models import generate_text_with_fallback, get_genai_client
from rag.hexnode_tools import search_hexnode_docs
from rag.keka_rag.rag_chain import get_llm
from rag.keka_rag.tools import search_keka_policies


logger = logging.getLogger(__name__)


class CombinedGraphState(TypedDict):
    question: str
    hexnode_result: Optional[Dict[str, Any]]
    keka_result: Optional[Dict[str, Any]]
    final_answer: str
    chunks: List[Dict[str, Any]]


@dataclass(frozen=True)
class CombinedGraphRuntime:
    graph: Any


def with_source_labels(chunks: List[Dict[str, Any]], source: str, label: str) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []

    for chunk in chunks:
        item = dict(chunk)
        item["kb_source"] = source
        item["kb_source_label"] = label
        result.append(item)

    return result


def format_result_section(title: str, result: Optional[Dict[str, Any]]) -> str:
    if not result:
        return f"{title} evidence:\nInformation not available."

    chunks = result.get("chunks") or []

    if chunks:
        chunk_lines = []
        for index, chunk in enumerate(chunks[:5], start=1):
            chunk_title = chunk.get("title", "Untitled")
            chunk_content = (chunk.get("content") or "").strip().replace("\n", " ")
            if len(chunk_content) > 900:
                chunk_content = f"{chunk_content[:900].rstrip()}..."
            chunk_lines.append(f"{index}. {chunk_title}: {chunk_content}")
        evidence = "\n".join(chunk_lines)
    else:
        evidence = "No supporting excerpts were retrieved."

    return f"{title} evidence:\n{evidence}"


def build_combined_graph_runtime(
    vector_store,
    keka_retriever,
) -> CombinedGraphRuntime:
    if StateGraph is None:
        raise ImportError("langgraph is not installed")

    graph = StateGraph(CombinedGraphState)

    def retrieve_hexnode(state: CombinedGraphState):
        return {
            "hexnode_result": search_hexnode_docs(
                state["question"],
                vector_store,
            )
        }

    def retrieve_keka(state: CombinedGraphState):
        return {
            "keka_result": search_keka_policies(
                state["question"],
                keka_retriever,
            )
        }

    def synthesize(state: CombinedGraphState):
        hexnode_result = state.get("hexnode_result") or {}
        keka_result = state.get("keka_result") or {}

        combined_chunks = with_source_labels(
            hexnode_result.get("chunks", []),
            "hexnode",
            "Hexnode Docs",
        ) + with_source_labels(
            keka_result.get("chunks", []),
            "keka",
            "Keka Policies",
        )

        prompt = ChatPromptTemplate.from_template(
            """
You are a combined enterprise assistant for Hexnode documentation and Keka HR policies.

Use ONLY the source-specific evidence provided below.

Rules:
- Keep the answer grounded in the provided material.
- Include a source only when it directly helps answer some part of the question.
- If both sources are useful, organize the answer with short markdown headings.
- If only one source is useful, answer with that source only.
- Do not add statements saying the other source does not contain the answer unless the user explicitly asks for a cross-source comparison.
- For multi-part questions, answer each part with the relevant source and do not force artificial overlap.
- Keep the answer concise and avoid repeating which source did not help.
- Do not invent links, policy details, or setup steps.
- End with a short combined takeaway only when it genuinely helps the user.

Question:
{question}

Hexnode material:
{hexnode_section}

Keka material:
{keka_section}

Answer:
"""
        )

        llm = get_llm()
        client = get_genai_client()
        rendered_prompt = prompt.format(
            question=state["question"],
            hexnode_section=format_result_section("Hexnode", hexnode_result),
            keka_section=format_result_section("Keka", keka_result),
        )
        try:
            response = llm.invoke(rendered_prompt)
            final_answer = content_to_text(response.content)
        except Exception as exc:
            logger.warning("Combined graph synthesize fallback triggered: %s", exc)
            final_answer = generate_text_with_fallback(client, rendered_prompt) or "Information not found."

        return {
            "final_answer": final_answer,
            "chunks": combined_chunks,
        }

    graph.add_node("retrieve_hexnode", retrieve_hexnode)
    graph.add_node("retrieve_keka", retrieve_keka)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "retrieve_hexnode")
    graph.add_edge(START, "retrieve_keka")
    graph.add_edge("retrieve_hexnode", "synthesize")
    graph.add_edge("retrieve_keka", "synthesize")
    graph.add_edge("synthesize", END)

    return CombinedGraphRuntime(graph=graph.compile())


def run_combined_graph(
    question: str,
    vector_store,
    keka_retriever,
    runtime: Optional[CombinedGraphRuntime] = None,
) -> Dict[str, Any]:
    active_runtime = runtime or build_combined_graph_runtime(
        vector_store,
        keka_retriever,
    )

    initial_state: CombinedGraphState = {
        "question": question,
        "hexnode_result": None,
        "keka_result": None,
        "final_answer": "",
        "chunks": [],
    }

    result = active_runtime.graph.invoke(initial_state)
    hexnode_result = result.get("hexnode_result") or {}
    keka_result = result.get("keka_result") or {}

    hexnode_calls = [
        {
            "source": "hexnode",
            **tool_call,
        }
        for tool_call in hexnode_result.get("tool_calls", [])
    ]
    keka_calls = [
        {
            "source": "keka",
            **tool_call,
        }
        for tool_call in keka_result.get("tool_calls", [])
    ]

    return {
        "question": question,
        "answer": result.get("final_answer", ""),
        "chunks": result.get("chunks", []),
        "tool_result": {
            "tool_name": "combined_hexnode_keka_graph",
            "question": question,
            "chunks": result.get("chunks", []),
            "hexnode_result": hexnode_result.get("tool_result", hexnode_result),
            "keka_result": keka_result.get("tool_result", keka_result),
        },
        "tool_calls": hexnode_calls + keka_calls,
        "source_results": {
            "hexnode": hexnode_result,
            "keka": keka_result,
        },
    }
