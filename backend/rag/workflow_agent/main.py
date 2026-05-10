import os
import time
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from .tools import build_workflow_tools

load_dotenv()


SYSTEM_PROMPT = """
You are a support workflow agent.

You have three tools:
- troubleshoot_issue_tool: use this for troubleshooting or support diagnosis.
- create_jira_ticket_tool: use this when the user wants to create or raise a Jira ticket.
- manage_jira_ticket_tool: use this when the user wants to close or delete a ticket.

Rules:
- If the issue is fixed, close the ticket instead of deleting it.
- If the user asks to delete, call manage_jira_ticket_tool with action `delete`.
- Deletion is Jira-approved: the tool will request approval first, wait if approval is pending, or delete after the Jira label delete-approved is added.
- If the user asks to create a Jira ticket and asks you to create the summary or description, use the conversation history to draft them.
- Do not ask for summary and description when the issue is already clear from recent user messages.
- For ticket descriptions, include the problem, what was tried, and the current status when available.
- Preserve useful Markdown formatting returned by tools, such as bold labels and bullet lists.
- Keep the final answer short and clear.
""".strip()


class WorkflowState(TypedDict):
    messages: List[Any]
    tool_calls: List[Dict[str, Any]]
    answer: str
    timings: Dict[str, Any]


def get_llm(tools):
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY1")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY1 before running the workflow agent.")

    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
        temperature=0,
        google_api_key=api_key,
    ).bind_tools(tools)


def should_continue(state: WorkflowState):
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "tools"

    return END


def build_workflow_graph(vector_store):
    tools = build_workflow_tools(vector_store)
    tools_by_name = {tool.name: tool for tool in tools}
    llm = get_llm(tools)

    graph = StateGraph(WorkflowState)

    def agent_node(state: WorkflowState) -> Dict[str, Any]:
        start = time.perf_counter()
        response = llm.invoke(state["messages"])
        elapsed = round(time.perf_counter() - start, 4)

        timings = dict(state.get("timings", {}))
        timings["agent_llm_seconds"] = round(timings.get("agent_llm_seconds", 0) + elapsed, 4)

        updates: Dict[str, Any] = {
            "messages": state["messages"] + [response],
            "timings": timings,
        }

        if not getattr(response, "tool_calls", None):
            updates["answer"] = response.content

        return updates

    def tool_node(state: WorkflowState) -> Dict[str, Any]:
        last_message = state["messages"][-1]
        new_messages = list(state["messages"])
        tool_calls = list(state["tool_calls"])
        timings = dict(state.get("timings", {}))
        tool_timings = list(timings.get("tools", []))

        for tool_call in last_message.tool_calls:
            selected_tool = tools_by_name[tool_call["name"]]
            start = time.perf_counter()
            tool_output = selected_tool.invoke(tool_call["args"])
            elapsed = round(time.perf_counter() - start, 4)

            timings["tool_execution_seconds"] = round(
                timings.get("tool_execution_seconds", 0) + elapsed,
                4,
            )
            tool_timings.append(
                {
                    "tool": tool_call["name"],
                    "seconds": elapsed,
                }
            )

            tool_calls.append(
                {
                    "tool": tool_call["name"],
                    "args": tool_call["args"],
                }
            )

            new_messages.append(
                ToolMessage(
                    content=tool_output,
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )

        return {
            "messages": new_messages,
            "tool_calls": tool_calls,
            "timings": {
                **timings,
                "tools": tool_timings,
            },
        }

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def build_messages(question: str, history=None) -> List[Any]:
    messages: List[Any] = [SystemMessage(content=SYSTEM_PROMPT)]

    for item in (history or [])[-5:]:
        text = (item.get("text") or "").strip()
        if not text:
            continue

        if item.get("role") == "ai":
            messages.append(AIMessage(content=text))
        else:
            messages.append(HumanMessage(content=text))

    messages.append(HumanMessage(content=question))
    return messages


def run_workflow_agent(question: str, vector_store=None, history=None) -> Dict[str, Any]:
    total_start = time.perf_counter()
    graph_runtime = build_workflow_graph(vector_store)
    result = graph_runtime.invoke(
        {
            "messages": build_messages(question, history),
            "tool_calls": [],
            "answer": "",
            "timings": {},
        }
    )
    timings = dict(result.get("timings", {}))
    timings["workflow_total_seconds"] = round(time.perf_counter() - total_start, 4)

    return {
        "question": question,
        "answer": result["answer"],
        "chunks": [],
        "tool_calls": result.get("tool_calls", []),
        "source": "workflow_agent",
        "timings": timings,
    }


if __name__ == "__main__":
    response = run_workflow_agent("I need to troubleshoot a VPN policy issue.")
    print(response["answer"])
