import asyncio
from langchain_core.tools import tool

from rag.hexnode_tools import get_hexnode_setup_steps
from postgres_mcp_server import (
    get_device_by_name,
    search_devices,
    get_compliance_summary,
)

from .jira_client import (
    JiraClientError,
    create_jira_ticket,
    close_jira_ticket,
    handle_delete_jira_ticket,
)


def run_async(coro):
    """Execute an async coroutine from synchronous LangChain tool callbacks."""
    return asyncio.run(coro)


def format_tool_error(action: str, exc: Exception) -> str:
    return (
        f"{action} failed.\n"
        f"Reason: {exc}\n"
        "Check the configured Jira project, issue type, credentials, and Jira field permissions."
    )


def build_troubleshoot_issue_tool(vector_store):
    @tool
    def troubleshoot_issue_tool(question: str) -> str:
        """Use this when the user wants troubleshooting help for a support issue."""
        if vector_store is None:
            return "Hexnode knowledge base is not available for troubleshooting right now."

        result = get_hexnode_setup_steps(question, vector_store)
        return result["formatted_context"]

    return troubleshoot_issue_tool


@tool
def create_jira_ticket_tool(summary: str, description: str) -> str:
    """Use this when the user wants to create or raise a Jira support ticket."""
    try:
        ticket = create_jira_ticket(summary, description)
    except (JiraClientError, ValueError) as exc:
        return format_tool_error("Jira ticket creation", exc)

    return (
        "**Jira ticket created.**\n"
        f"- **Ticket:** {ticket['issue_key']}\n"
        f"- **URL:** {ticket['url']}"
    )


@tool
def manage_jira_ticket_tool(issue_key: str, action: str) -> str:
    """Use this when the user wants to close or delete a Jira ticket."""
    if action == "close":
        try:
            ticket = close_jira_ticket(issue_key)
        except (JiraClientError, ValueError) as exc:
            return format_tool_error(f"Jira ticket {issue_key} close", exc)

        return (
            f"Jira ticket {issue_key} closed.\n"
            f"Transition: {ticket['transition']}"
        )

    elif action == "delete":
        try:
            ticket = handle_delete_jira_ticket(issue_key)
        except (JiraClientError, ValueError) as exc:
            return format_tool_error(f"Jira ticket {issue_key} delete", exc)

        if ticket["status"] == "pending_delete_approval":
            return (
                "**Delete approval requested.**\n"
                f"- **Ticket:** {ticket['issue_key']}\n"
                "- **Status:** Waiting for team approval in Jira.\n"
                "- Once approved, the ticket will be deleted automatically."
            )

        reason = ticket.get("reason", "Ticket deleted successfully.")
        return (
            f"Delete check for Jira ticket {ticket['issue_key']}.\n"
            f"Status: {ticket['status']}\n"
            f"Reason: {reason}"
        )

    return f"unsupported action {action} for jira ticket {issue_key}"


@tool
def get_device_details_tool(device_name: str) -> str:
    """Use this tool to look up specifications, owner email, OS build, and compliance status for a device."""
    return run_async(get_device_by_name(device_name))


@tool
def search_devices_tool(user_email: str = None, os_type: str = None, compliance_status: str = None) -> str:
    """Use this tool to search or filter device inventory by owner email, OS (Windows/macOS/iOS/Android), or compliance status."""
    return run_async(search_devices(user_email=user_email, os_type=os_type, compliance_status=compliance_status))


@tool
def get_device_summary_tool() -> str:
    """Use this tool to get device inventory metrics (total devices, compliant count, non-compliant count, compliance rate)."""
    return run_async(get_compliance_summary())


def build_workflow_tools(vector_store):
    return [
        build_troubleshoot_issue_tool(vector_store),
        create_jira_ticket_tool,
        manage_jira_ticket_tool,
        get_device_details_tool,
        search_devices_tool,
        get_device_summary_tool,
    ]
