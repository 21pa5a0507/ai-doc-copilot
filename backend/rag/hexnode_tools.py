from rag.embendings import get_embending as embed_text
from rag.answer_generator import generate_answer


def format_hexnode_chunks(chunks):
    """
    Convert retrieved chunks into a clean text block the LLM can read.
    """
    if not chunks:
        return "No relevant Hexnode documentation was found."

    formatted_parts = []

    for index, chunk in enumerate(chunks, start=1):
        title = chunk.get("title", "Untitled")
        source = chunk.get("source", "unknown")
        content = chunk.get("content", "").strip()

        formatted_parts.append(
            f"Result {index}\n"
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Content: {content}"
        )

    return "\n\n".join(formatted_parts)


def list_hexnode_topics(vector_store):
    """
    Return the distinct titles/topics available in the Hexnode vector store.
    """
    seen = set()
    topics = []

    for chunk in getattr(vector_store, "chunks", []):
        title = (chunk.get("title") or "").strip()
        if not title or title in seen:
            continue
        seen.add(title)
        topics.append(title)

    topics.sort()
    preview_topics = topics[:60]

    if preview_topics:
        topic_chunks = [
            {
                "title": "Hexnode Topics",
                "content": "\n".join(preview_topics),
                "source": "hexnode_topics",
            }
        ]
        formatted_context = generate_answer(
            "Summarize these Hexnode documentation topics into minimal categories.",
            topic_chunks,
            mode="topics",
        )
    else:
        formatted_context = "No Hexnode topics are available in the vector store."

    return {
        "tool_name": "list_hexnode_topics",
        "formatted_context": formatted_context,
    }


def search_hexnode_docs(question, vector_store):
    """
    Search the existing Hexnode/general docs vector store and return
    both structured chunks and formatted context.
    """
    query_embedding = embed_text(question)
    chunks = vector_store.search(query_embedding, question)
    formatted_context = format_hexnode_chunks(chunks)

    return {
        "tool_name": "search_hexnode_docs",
        "question": question,
        "chunks": chunks,
        "formatted_context": formatted_context,
    }


def get_hexnode_setup_steps(question, vector_store):
    """
    Return likely setup or enrollment steps grounded in retrieved Hexnode docs.
    """
    search_result = search_hexnode_docs(question, vector_store)
    chunks = search_result["chunks"]

    if not chunks:
        formatted_context = (
            "No relevant Hexnode documentation was found for this query. "
            "Provide the best direct guidance you can based on the question."
        )
    else:
        formatted_context = generate_answer(question, chunks, mode="steps")

    return {
        "tool_name": "get_hexnode_setup_steps",
        "question": question,
        "chunks": chunks,
        "formatted_context": formatted_context,
    }


def handle_hexnode_question(question, vector_store, answer_generator, graph_runtime=None):
    """
    Handle the full Hexnode/default question flow and return the API response payload.
    """
    try:
        from rag.hexnode_graph import run_hexnode_graph

        return run_hexnode_graph(question, vector_store, runtime=graph_runtime)
    except Exception as exc:
        print(f"Hexnode graph fallback triggered: {exc}")

    lowered_question = question.lower()

    topic_keywords = (
        "list topics",
        "what topics",
        "available topics",
        "available docs",
        "available documentation",
        "what documentation",
        "show topics",
    )
    setup_keywords = (
        "how to",
        "how do i",
        "setup",
        "set up",
        "configure",
        "configuration",
        "enroll",
        "enrollment",
        "install",
        "enable",
        "disable",
        "assign",
        "create policy",
        "step",
        "steps",
        "step-by-step",
        "step by step",
        "navigate",
        "where do i",
        "workflow",
        "procedure",
        "walkthrough",
    )

    if any(keyword in lowered_question for keyword in topic_keywords):
        tool_result = list_hexnode_topics(vector_store)
        chunks = []
        answer = tool_result["formatted_context"]
    elif any(keyword in lowered_question for keyword in setup_keywords):
        tool_result = get_hexnode_setup_steps(question, vector_store)
        chunks = tool_result["chunks"]
        answer = tool_result["formatted_context"]
    else:
        tool_result = search_hexnode_docs(question, vector_store)
        chunks = tool_result["chunks"]
        answer = answer_generator(question, chunks)

    print(f"Retrieved {len(chunks)} chunks for question: {question}")
    print(f"Tool used: {tool_result['tool_name']}")

    return {
        "question": question,
        "chunks": chunks,
        "answer": answer,
        "tool_result": tool_result,
        "tool_calls": [],
    }
