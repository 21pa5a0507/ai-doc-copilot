import json
import sys

from rag.workflow_agent.jira_client import JiraClientError, process_delete_approvals


def main() -> int:
    try:
        result = process_delete_approvals()
    except (JiraClientError, ValueError) as exc:
        print(f"Delete approval run failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
