import os

import requests
from dotenv import load_dotenv

from config.paths import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")
DELETE_REQUESTED_LABEL = "delete-requested"
DELETE_APPROVED_LABEL = "delete-approved"
DELETE_REQUEST_COMMENT = (
    "Deletion requested by workflow agent. "
    "To approve deletion, add the label delete-approved to this Jira issue."
)


def get_jira_credentials() -> dict:
    return {
        "base_url": os.getenv("JIRA_BASE_URL"),
        "email": os.getenv("JIRA_EMAIL"),
        "api_token": os.getenv("JIRA_API_TOKEN"),
        "project_key": os.getenv("JIRA_PROJECT_KEY"),
        "issue_type": os.getenv("JIRA_ISSUE_TYPE"),
    }


def validate_jira_credentials(creds: dict) -> None:
    required_keys = ["base_url", "email", "api_token", "project_key", "issue_type"]
    missing_keys = [key for key in required_keys if not creds.get(key)]
    if missing_keys:
        raise ValueError(f"Missing Jira credentials: {', '.join(missing_keys)}")


def create_jira_ticket(summary: str, description: str) -> dict:
    creds = get_jira_credentials()
    validate_jira_credentials(creds)

    url = f"{creds['base_url'].rstrip('/')}/rest/api/3/issue"
    payload = {
        "fields": {
            "project": {
                "key": creds["project_key"],
            },
            "summary": summary,
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": description,
                            }
                        ],
                    }
                ],
            },
            "issuetype": {
                "name": creds["issue_type"],
            },
        }
    }

    response = requests.post(
        url,
        json=payload,
        auth=(creds["email"], creds["api_token"]),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()

    return {
        "issue_key": data["key"],
        "issue_id": data["id"],
        "url": f"{creds['base_url'].rstrip('/')}/browse/{data['key']}",
    }


def get_jira_transitions(issue_key: str) -> list:
    creds = get_jira_credentials()
    validate_jira_credentials(creds)

    url = f"{creds['base_url'].rstrip('/')}/rest/api/3/issue/{issue_key}/transitions"

    response = requests.get(
        url,
        auth=(creds["email"], creds["api_token"]),
        headers={
            "Accept": "application/json",
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()

    return data.get("transitions", [])


def close_jira_ticket(issue_key: str) -> dict:
    creds = get_jira_credentials()
    validate_jira_credentials(creds)

    transitions = get_jira_transitions(issue_key)

    close_transition = None
    for transition in transitions:
        if transition.get("name") == "Close":
            close_transition = transition
            break

    if close_transition is None:
        available = [transition.get("name") for transition in transitions]
        raise ValueError(f"Close transition not found. Available transitions: {available}")

    url = f"{creds['base_url'].rstrip('/')}/rest/api/3/issue/{issue_key}/transitions"

    response = requests.post(
        url,
        json={
            "transition": {
                "id": close_transition["id"],
            }
        },
        auth=(creds["email"], creds["api_token"]),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        timeout=20,
    )

    response.raise_for_status()

    return {
        "issue_key": issue_key,
        "status": "closed",
        "transition": "Close",
    }


def add_jira_label(issue_key: str, label: str) -> dict:
    creds = get_jira_credentials()
    validate_jira_credentials(creds)

    url = f"{creds['base_url'].rstrip('/')}/rest/api/3/issue/{issue_key}"

    response = requests.put(
        url,
        json={
            "update": {
                "labels": [
                    {"add": label}
                ]
            }
        },
        auth=(creds["email"], creds["api_token"]),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        timeout=20,
    )

    response.raise_for_status()

    return {
        "issue_key": issue_key,
        "added_label": label,
        "status": "label_added",
    }


def add_jira_comment(issue_key: str, comment: str) -> dict:
    creds = get_jira_credentials()
    validate_jira_credentials(creds)

    url = f"{creds['base_url'].rstrip('/')}/rest/api/3/issue/{issue_key}/comment"

    response = requests.post(
        url,
        json={
            "body": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": comment,
                            }
                        ],
                    }
                ],
            }
        },
        auth=(creds["email"], creds["api_token"]),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        timeout=20,
    )

    response.raise_for_status()
    data = response.json()

    return {
        "issue_key": issue_key,
        "comment_id": data["id"],
        "status": "comment_added",
    }


def request_delete_jira_ticket(issue_key: str) -> dict:
    label_result = add_jira_label(issue_key, DELETE_REQUESTED_LABEL)
    comment_result = add_jira_comment(issue_key, DELETE_REQUEST_COMMENT)

    return {
        "issue_key": issue_key,
        "status": "pending_delete_approval",
        "requested_label": label_result["added_label"],
        "comment_id": comment_result["comment_id"],
        "approval_label": DELETE_APPROVED_LABEL,
    }


def delete_jira_ticket(issue_key: str) -> dict:
    creds = get_jira_credentials()
    validate_jira_credentials(creds)

    url = f"{creds['base_url'].rstrip('/')}/rest/api/3/issue/{issue_key}"

    response = requests.delete(
        url,
        auth=(creds["email"], creds["api_token"]),
        headers={
            "Accept": "application/json",
        },
        timeout=20,
    )

    response.raise_for_status()

    return {
        "issue_key": issue_key,
        "status": "deleted",
    }


def get_jira_issue(issue_key: str) -> dict:
    creds = get_jira_credentials()
    validate_jira_credentials(creds)

    url = f"{creds['base_url'].rstrip('/')}/rest/api/3/issue/{issue_key}"

    response = requests.get(
        url,
        auth=(creds["email"], creds["api_token"]),
        headers={
            "Accept": "application/json",
        },
        timeout=20,
    )

    response.raise_for_status()

    return response.json()


def handle_delete_jira_ticket(issue_key: str) -> dict:
    issue = get_jira_issue(issue_key)
    labels = issue.get("fields", {}).get("labels", [])

    if DELETE_REQUESTED_LABEL not in labels:
        return request_delete_jira_ticket(issue_key)

    if DELETE_APPROVED_LABEL not in labels:
        return {
            "issue_key": issue_key,
            "status": "pending_approval",
            "reason": "Delete is waiting for Jira approval. Add label delete-approved to approve.",
            "labels": labels,
        }

    return delete_jira_ticket(issue_key)


def find_delete_requested_tickets() -> list:
    creds = get_jira_credentials()
    validate_jira_credentials(creds)

    url = f"{creds['base_url'].rstrip('/')}/rest/api/3/search/jql"

    jql = (
        f'project = {creds["project_key"]} '
        f'AND labels = "{DELETE_REQUESTED_LABEL}"'
    )

    response = requests.get(
        url,
        params={
            "jql": jql,
            "fields": "labels",
            "maxResults": 50,
        },
        auth=(creds["email"], creds["api_token"]),
        headers={
            "Accept": "application/json",
        },
        timeout=20,
    )

    response.raise_for_status()
    data = response.json()

    return data.get("issues", [])


def process_delete_approvals() -> dict:
    issues = find_delete_requested_tickets()

    deleted = []
    pending = []

    for issue in issues:
        issue_key = issue["key"]
        labels = issue.get("fields", {}).get("labels", [])

        if DELETE_APPROVED_LABEL in labels:
            result = delete_jira_ticket(issue_key)
            deleted.append(result)
        else:
            pending.append(
                {
                    "issue_key": issue_key,
                    "status": "pending_approval",
                    "labels": labels,
                }
            )

    return {
        "checked": len(issues),
        "deleted": deleted,
        "pending": pending,
    }
