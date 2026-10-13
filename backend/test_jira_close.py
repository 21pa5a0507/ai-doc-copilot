import unittest
from unittest.mock import Mock, patch

from rag.workflow_agent import jira_client


class JiraCloseTests(unittest.TestCase):
    def test_transition_selection_and_default(self):
        for settings, name in [({}, "Close"), ({"JIRA_CLOSE_TRANSITION": "Done"}, "Done")]:
            with (
                patch.dict(jira_client.os.environ, settings, clear=True),
                patch.object(jira_client, "get_jira_credentials", return_value={
                    "base_url": "https://example.atlassian.net", "email": "test@example.com",
                    "api_token": "test", "project_key": "TEST", "issue_type": "Task",
                }),
                patch.object(jira_client, "get_jira_transitions", return_value=[
                    {"id": "11", "name": "In Progress"}, {"id": "31", "name": name},
                ]),
                patch.object(jira_client, "jira_request", return_value=Mock(ok=True)) as request,
            ):
                result = jira_client.close_jira_ticket("TEST-1")
                self.assertEqual(result["transition"], name)
                self.assertEqual(request.call_args.kwargs["json"], {"transition": {"id": "31"}})

    def test_missing_transition_does_not_modify_ticket(self):
        with (
            patch.dict(jira_client.os.environ, {"JIRA_CLOSE_TRANSITION": "Done"}),
            patch.object(jira_client, "get_jira_credentials", return_value={}),
            patch.object(jira_client, "validate_jira_credentials"),
            patch.object(jira_client, "get_jira_transitions", return_value=[{"name": "In Progress"}]),
            patch.object(jira_client, "jira_request") as request,
        ):
            with self.assertRaisesRegex(ValueError, "'Done' not found"):
                jira_client.close_jira_ticket("TEST-1")
            request.assert_not_called()
