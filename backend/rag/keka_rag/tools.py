import logging
import os
import time

from langchain_core.prompts import ChatPromptTemplate

from rag.gemini_models import generate_text_with_fallback, get_genai_client
from rag.keka_rag.rag_chain import get_llm


logger = logging.getLogger(__name__)


def should_use_agent_router():
    return os.getenv("USE_KEKA_AGENT_ROUTER", "false").lower() in {"1", "true", "yes", "on"}


def format_keka_chunks(docs):
    if not docs:
        return "No relevant Keka policy documents were found."

    formatted_parts = []

    for index, doc in enumerate(docs, start=1):
        title = doc.metadata.get("file_name", "Keka document")
        source = doc.metadata.get("source", "keka")
        content = doc.page_content.strip()

        formatted_parts.append(
            f"Result {index}\n"
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Content: {content}"
        )

    return "\n\n".join(formatted_parts)


def docs_to_chunks(docs):
    return [
        {
            "title": doc.metadata.get("file_name", "Keka document"),
            "content": doc.page_content,
            "source": doc.metadata.get("source", "keka"),
        }
        for doc in docs
    ]


def collect_policy_names(retriever):
    seen = set()
    names = []

    for doc in getattr(retriever, "docs", []):
        name = doc.metadata.get("file_name", "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)

    names.sort()
    return names


def search_keka_policies(question, retriever):
    docs = retriever.invoke(question)
    formatted_context = format_keka_chunks(docs)
    chunks = docs_to_chunks(docs)
    timings = getattr(retriever, "last_timings", {})

    return {
        "tool_name": "search_keka_policies",
        "question": question,
        "chunks": chunks,
        "formatted_context": formatted_context,
        "timings": timings,
    }


def list_keka_policies(retriever):
    policy_names = collect_policy_names(retriever)
    formatted_context = "\n".join(f"- {name}" for name in policy_names)

    if not formatted_context:
        formatted_context = "No Keka policy documents are available."

    return {
        "tool_name": "list_keka_policies",
        "formatted_context": formatted_context,
        "policies": policy_names,
    }


def get_keka_process_steps(question, retriever):
    docs = retriever.invoke(question)
    formatted_context = format_keka_chunks(docs)
    chunks = docs_to_chunks(docs)
    timings = dict(getattr(retriever, "last_timings", {}))

    prompt = ChatPromptTemplate.from_template(
        """
You are an expert assistant for Keka HR policies.

Use ONLY the provided context.

Rules:
- If the context contains a clear employee or HR process, return short numbered steps.
- If the context does not contain a clear process, return a short direct answer from the context instead.
- Do NOT invent missing steps or policy details.
- Remove repetition and overlapping details.
- Do NOT mention the context, documents, or tool in the answer.
- If returning steps, return only the numbered steps.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    llm = get_llm()
    client = get_genai_client()
    rendered_prompt = prompt.format(
        context=formatted_context,
        question=question,
    )
    try:
        answer_start = time.perf_counter()
        response = llm.invoke(rendered_prompt)
        answer = response.content
        timings["answer_seconds"] = round(time.perf_counter() - answer_start, 4)
    except Exception as exc:
        logger.warning("Keka process-steps fallback triggered: %s", exc)
        answer_start = time.perf_counter()
        answer = generate_text_with_fallback(client, rendered_prompt) or "I don't know"
        timings["answer_seconds"] = round(time.perf_counter() - answer_start, 4)

    return {
        "tool_name": "get_keka_process_steps",
        "question": question,
        "chunks": chunks,
        "formatted_context": answer,
        "timings": timings,
    }


def handle_keka_question(question, retriever, rag_chain, agent=None):
    lowered_question = question.lower()

    list_keywords = (
        "list policies",
        "what policies",
        "available policies",
        "available documents",
        "available docs",
        "show policies",
        "policy documents",
    )
    process_keywords = (
        "how to",
        "how do i",
        "apply",
        "request",
        "claim",
        "submit",
        "raise",
        "reimburse",
        "reimbursement",
        "process",
        "procedure",
        "workflow",
        "step",
        "steps",
        "step-by-step",
        "step by step",
    )

    if any(keyword in lowered_question for keyword in list_keywords):
        tool_result = list_keka_policies(retriever)
        chunks = []
        answer = tool_result["formatted_context"]
    elif any(keyword in lowered_question for keyword in process_keywords):
        tool_result = get_keka_process_steps(question, retriever)
        chunks = tool_result["chunks"]
        answer = tool_result["formatted_context"]
    elif should_use_agent_router():
        try:
            from rag.keka_rag.keka_agent import build_keka_agent, run_keka_agent

            active_agent = agent or build_keka_agent(retriever)
            return run_keka_agent(question, active_agent, rag_chain)
        except Exception as exc:
            logger.warning("Keka agent fallback triggered: %s", exc)
            tool_result = search_keka_policies(question, retriever)
            chunks = tool_result["chunks"]
            answer_start = time.perf_counter()
            answer = answer_keka_question(question, rag_chain, tool_result)
            tool_result["timings"]["answer_seconds"] = round(time.perf_counter() - answer_start, 4)
    else:
        tool_result = search_keka_policies(question, retriever)
        chunks = tool_result["chunks"]
        answer_start = time.perf_counter()
        answer = answer_keka_question(question, rag_chain, tool_result)
        tool_result["timings"]["answer_seconds"] = round(time.perf_counter() - answer_start, 4)

    logger.info("Keka retrieval used %s and returned %s docs", tool_result["tool_name"], len(chunks))

    return {
        "question": question,
        "chunks": chunks,
        "answer": answer,
        "tool_result": tool_result,
        "tool_calls": [],
        "timings": tool_result.get("timings", {}),
    }


def answer_keka_question(question, rag_chain, tool_result):
    context = tool_result.get("formatted_context", "")

    if hasattr(rag_chain, "answer_with_context"):
        return rag_chain.answer_with_context(question, context)

    logger.warning("Keka rag_chain has no answer_with_context helper; falling back to full chain.")
    return rag_chain(question, debug=False)
