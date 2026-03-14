"""Tests for judge escalation rules -- subtle axis triggers."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock
from autotrust.config import load_spec


@pytest.fixture
def spec():
    return load_spec(Path(__file__).parent.parent / "spec.yaml")


def test_judge_escalation_subtle_axis_above_threshold(spec):
    """Subtle axis score > escalate_threshold triggers judge fallback."""
    from autotrust.eval import run_judge_fallback

    fast_scores = {a.name: 0.1 for a in spec.trust_axes}
    fast_scores["deceit"] = 0.8  # deceit is in axis_groups.subtle, above threshold 0.6

    mock_judge = MagicMock()
    mock_judge.judge.return_value = {"deceit": 0.9}

    chain = MagicMock()
    run_judge_fallback(chain, fast_scores, mock_judge, spec)
    # Judge should have been called for subtle axes
    mock_judge.judge.assert_called_once()


def test_judge_escalation_fast_axis_no_escalation(spec):
    """Fast axis high score does not trigger escalation."""
    from autotrust.eval import run_judge_fallback

    fast_scores = {a.name: 0.1 for a in spec.trust_axes}
    fast_scores["phish"] = 0.9  # phish is in fast group, not subtle

    mock_judge = MagicMock()
    chain = MagicMock()
    run_judge_fallback(chain, fast_scores, mock_judge, spec)
    # Judge should NOT have been called (no subtle axes above threshold)
    mock_judge.judge.assert_not_called()


def test_judge_escalation_below_threshold(spec):
    """Subtle axis below threshold does not trigger."""
    from autotrust.eval import run_judge_fallback

    fast_scores = {a.name: 0.1 for a in spec.trust_axes}
    fast_scores["deceit"] = 0.3  # below escalate_threshold 0.6

    mock_judge = MagicMock()
    chain = MagicMock()
    run_judge_fallback(chain, fast_scores, mock_judge, spec)
    mock_judge.judge.assert_not_called()
