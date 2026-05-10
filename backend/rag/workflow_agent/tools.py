from langchain_core.tools import tool

from rag.hexnode_tools import get_hexnode_setup_steps

from .jira_client import (
    create_jira_ticket,
    close_jira_ticket,
    handle_delete_jira_ticket,
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
    ticket = create_jira_ticket(summary, description)
    return (
        "**Jira ticket created.**\n"
        f"- **Ticket:** {ticket['issue_key']}\n"
        f"- **URL:** {ticket['url']}"
    )


@tool
def manage_jira_ticket_tool(issue_key: str, action: str) -> str:
    """Use this when the user wants to close or delete a Jira ticket."""
    if action == "close":
        ticket = close_jira_ticket(issue_key)
        return (
            f"Jira ticket {issue_key} closed.\n"
            f"Transition: {ticket['transition']}"
        )

    elif action == "delete":
        ticket = handle_delete_jira_ticket(issue_key)

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


def build_workflow_tools(vector_store):
    return [
        build_troubleshoot_issue_tool(vector_store),
        create_jira_ticket_tool,
        manage_jira_ticket_tool,
    ]
