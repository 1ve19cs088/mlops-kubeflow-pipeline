"""
Tests for dashboard/deployment_actions.py — the placeholder registry
describing what promote/archive/rollback/deploy would do.
"""

from dashboard.deployment_actions import ACTIONS, get_action


def test_get_action_returns_known_actions():
    for key in ("promote", "archive", "rollback", "deploy"):
        action = get_action(key)
        assert action is not None
        assert action.key == key
        assert action.label
        assert action.description
        assert action.future_stage


def test_get_action_returns_none_for_unknown_key():
    assert get_action("does-not-exist") is None


def test_all_actions_have_distinct_labels():
    labels = [action.label for action in ACTIONS.values()]
    assert len(labels) == len(set(labels))
