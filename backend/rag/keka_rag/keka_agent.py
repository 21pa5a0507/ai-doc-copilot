from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from rag.content import content_to_text
from rag.keka_rag.rag_chain import get_llm
from rag.keka_rag.tools import (
    get_keka_process_steps,
    list_keka_policies,
    search_keka_policies,
)


SYSTEM_PROMPT = """
You are a Keka HR policy assistant.

Use the available tools to answer the user's question.

Tool guidance:
- Use `list_keka_policies_tool` when the user asks what policies or documents are available.
- Use `get_keka_process_steps_tool` when the user asks how to apply, request, claim, submit, or complete a Keka-related HR process.
- Use `search_keka_policies_tool` for policy rules, definitions, eligibility, limits, benefits, and general HR documentation questions.

Rules:
- Use only the tool results for the final answer.
- Keep the answer concise and clear.
- If the answer is not supported by the tool results, say so plainly.
""".strip()


_REQUEST_STATE: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "keka_agent_request_state",
    default=None,
)


def _store_tool_result(result: Dict[str, Any], tool_name: str, args: Dict[str, Any]) -> None:
    request_state = _REQUEST_STATE.get()
    if request_state is None:
        return

    request_state["latest_tool_result"] = dict(result)
    request_state["tool_trace"].append(
        {
            "tool": tool_name,
            "args": dict(args),
        }
    )


@contextmanager
def _request_state_scope(question: str):
    request_state: Dict[str, Any] = {
        "latest_tool_result": {
            "tool_name": None,
            "question": question,
            "chunks": [],
            "formatted_context": "",
        },
        "tool_trace": [],
    }
    token = _REQUEST_STATE.set(request_state)

    try:
        yield request_state
    finally:
        _REQUEST_STATE.reset(token)


def build_keka_agent(retriever):
    @tool
    def search_keka_policies_tool(question: str) -> str:
        """Search Keka policies for policy rules, eligibility, leave, insurance, benefits, and HR information."""
        result = search_keka_policies(question, retriever)
        _store_tool_result(result, result["tool_name"], {"question": question})
        return result["formatted_context"]

    @tool
    def list_keka_policies_tool() -> str:
        """List the available Keka policy document names."""
        result = list_keka_policies(retriever)
        _store_tool_result(result, result["tool_name"], {})
        return result["formatted_context"]

    @tool
    def get_keka_process_steps_tool(question: str) -> str:
        """Get Keka HR process steps for applying, requesting, claiming, or submitting something."""
        result = get_keka_process_steps(question, retriever)
        _store_tool_result(result, result["tool_name"], {"question": question})
        return result["formatted_context"]

    return create_agent(
        model=get_llm(),
        tools=[
            search_keka_policies_tool,
            list_keka_policies_tool,
            get_keka_process_steps_tool,
        ],
        system_prompt=SYSTEM_PROMPT,
        debug=False,
        name="keka-agent",
    )


def run_keka_agent(question: str, agent, rag_chain) -> Dict[str, Any]:
    with _request_state_scope(question) as request_state:
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )

    final_answer = ""
    for message in reversed(result.get("messages", [])):
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            final_answer = content_to_text(message.content)
            break

    latest_tool_result = dict(request_state["latest_tool_result"])
    tool_trace: List[Dict[str, Any]] = list(request_state["tool_trace"])

    if not final_answer:
        context = latest_tool_result.get("formatted_context", "")
        if context and hasattr(rag_chain, "answer_with_context"):
            final_answer = rag_chain.answer_with_context(question, context)
        else:
            final_answer = rag_chain(question, debug=False)

    return {
        "question": question,
        "chunks": latest_tool_result.get("chunks", []),
        "answer": final_answer,
        "tool_result": latest_tool_result,
        "tool_calls": tool_trace,
    }
