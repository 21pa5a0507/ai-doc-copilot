from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - handled at runtime if langgraph is missing
    END = START = StateGraph = None

from rag.hexnode_tools import (
    get_hexnode_setup_steps,
    list_hexnode_topics,
    search_hexnode_docs,
)
from rag.content import content_to_text
from rag.gemini_models import PRIMARY_MODEL, get_google_api_key

load_dotenv()


SYSTEM_PROMPT = """
You are a Hexnode documentation assistant.

Use the available tools to answer the user's question.

Tool guidance:
- Use `list_hexnode_topics_tool` when the user asks what documentation topics or categories are available.
- Use `get_hexnode_setup_steps_tool` when the user asks for step-by-step instructions, UI navigation paths, setup flows, enrollment flows, configuration sequences, or any procedural guidance.
- Use `search_hexnode_docs_tool` for policy questions, certificate questions, feature explanations, product behavior, conceptual questions, and any question where the user wants general information rather than explicit actions.
- Prefer `get_hexnode_setup_steps_tool` when the question contains words like "how to", "set up", "configure", "install", "enable", "disable", "step", "steps", "navigate", "where do I", "workflow", "procedure", or "walkthrough".

Rules:
- Use only the tool results for the final answer.
- Keep the answer concise and clear.
- If the tool results do not contain the answer, say so plainly.
""".strip()


class HexnodeGraphState(TypedDict):
    question: str
    messages: List[Any]
    tool_calls: List[Dict[str, Any]]
    tool_result: Optional[Dict[str, Any]]
    final_answer: str


@dataclass(frozen=True)
class HexnodeGraphRuntime:
    graph: Any
    available_tools: List[str]


@lru_cache(maxsize=1)
def _get_llm():
    return ChatGoogleGenerativeAI(
        model=PRIMARY_MODEL,
        temperature=0.2,
        google_api_key=get_google_api_key(),
    )


def build_hexnode_graph_runtime(vector_store) -> HexnodeGraphRuntime:
    if StateGraph is None:
        raise ImportError("langgraph is not installed")

    def run_search_tool(args: Dict[str, Any]) -> Dict[str, Any]:
        return search_hexnode_docs(args["question"], vector_store)

    def run_topics_tool(_args: Dict[str, Any]) -> Dict[str, Any]:
        return list_hexnode_topics(vector_store)

    def run_steps_tool(args: Dict[str, Any]) -> Dict[str, Any]:
        return get_hexnode_setup_steps(args["question"], vector_store)

    @tool
    def search_hexnode_docs_tool(question: str) -> str:
        """Search Hexnode documentation for policies, certificates, features, explanations, and general product questions."""
        return run_search_tool({"question": question})["formatted_context"]

    @tool
    def list_hexnode_topics_tool() -> str:
        """List the available Hexnode documentation topics."""
        return run_topics_tool({})["formatted_context"]

    @tool
    def get_hexnode_setup_steps_tool(question: str) -> str:
        """Get step-by-step setup, installation, enrollment, navigation, or configuration instructions from Hexnode docs."""
        return run_steps_tool({"question": question})["formatted_context"]

    tools = (
        search_hexnode_docs_tool,
        list_hexnode_topics_tool,
        get_hexnode_setup_steps_tool,
    )
    tool_handlers = {
        "search_hexnode_docs_tool": run_search_tool,
        "list_hexnode_topics_tool": run_topics_tool,
        "get_hexnode_setup_steps_tool": run_steps_tool,
    }
    llm = _get_llm().bind_tools(tools)

    graph = StateGraph(HexnodeGraphState)

    def agent_node(state: HexnodeGraphState):
        response = llm.invoke(state["messages"])
        updates: Dict[str, Any] = {
            "messages": state["messages"] + [response],
        }

        if not getattr(response, "tool_calls", None):
            updates["final_answer"] = content_to_text(response.content)

        return updates

    def tools_node(state: HexnodeGraphState):
        last_message = state["messages"][-1]
        new_messages = list(state["messages"])
        tool_trace = list(state["tool_calls"])
        latest_tool_result = state.get("tool_result")

        for tool_call in getattr(last_message, "tool_calls", []):
            handler = tool_handlers.get(tool_call["name"])
            if handler is None:
                raise ValueError(f"Unsupported Hexnode tool: {tool_call['name']}")

            result = handler(tool_call["args"])
            latest_tool_result = result
            tool_trace.append(
                {
                    "tool": result["tool_name"],
                    "args": tool_call["args"],
                }
            )
            new_messages.append(
                ToolMessage(
                    content=result["formatted_context"],
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )

        return {
            "messages": new_messages,
            "tool_calls": tool_trace,
            "tool_result": latest_tool_result,
        }

    def should_continue(state: HexnodeGraphState):
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return HexnodeGraphRuntime(
        graph=graph.compile(),
        available_tools=[tool_item.name for tool_item in tools],
    )


def run_hexnode_graph(question: str, vector_store, runtime: Optional[HexnodeGraphRuntime] = None):
    active_runtime = runtime or build_hexnode_graph_runtime(vector_store)

    initial_state: HexnodeGraphState = {
        "question": question,
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question),
        ],
        "tool_calls": [],
        "tool_result": None,
        "final_answer": "",
    }

    result = active_runtime.graph.invoke(initial_state)
    tool_result = result.get("tool_result") or {
        "tool_name": None,
        "question": question,
        "chunks": [],
        "formatted_context": "",
    }

    return {
        "question": question,
        "chunks": tool_result.get("chunks", []),
        "answer": result.get("final_answer", ""),
        "tool_result": tool_result,
        "tool_calls": result.get("tool_calls", []),
        "available_tools": list(active_runtime.available_tools),
    }
