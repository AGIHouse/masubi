"""Tests for Stage CLI, auto-transition, and Stage 2 subprocess mode in run_loop.py."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from autotrust.config import load_spec


@pytest.fixture
def spec():
    return load_spec(Path(__file__).parent.parent / "spec.yaml")


def test_cli_stage_argument():
    """Parsing --stage train sets stage to 'train'."""
    from run_loop import _parse_args
    args = _parse_args(["--stage", "train"])
    assert args.stage == "train"


def test_cli_default_stage():
    """No --stage defaults to 'prompt'."""
    from run_loop import _parse_args
    args = _parse_args([])
    assert args.stage == "prompt"


def test_auto_transition_triggers(spec):
    """After 3 consecutive no-improvement with stage='prompt', triggers transition."""
    from run_loop import _should_auto_transition
    assert _should_auto_transition("prompt", 3) is True
    assert _should_auto_transition("prompt", 2) is False
    assert _should_auto_transition("train", 3) is False


def test_auto_transition_calls_freeze(spec, tmp_path):
    """Auto-transition calls freeze_teacher()."""
    from run_loop import _auto_transition

    with patch("autotrust.freeze.freeze_teacher") as mock_freeze:
        mock_freeze.return_value = MagicMock()
        new_stage = _auto_transition(spec)
        mock_freeze.assert_called_once()
        assert new_stage == "train"


def test_stage2_time_limit(spec):
    """Stage 2 uses stage2_experiment_minutes from spec."""
    from run_loop import _get_time_limit
    limit = _get_time_limit(spec, "train")
    assert limit == spec.limits.stage2_experiment_minutes


def test_stage1_time_limit(spec):
    """Stage 1 uses stage1_experiment_minutes (or experiment_minutes fallback)."""
    from run_loop import _get_time_limit
    limit = _get_time_limit(spec, "prompt")
    expected = spec.limits.stage1_experiment_minutes or spec.limits.experiment_minutes
    assert limit == expected


def test_manual_stage_train_skips_stage1():
    """--stage train goes directly to Stage 2."""
    from run_loop import _parse_args
    args = _parse_args(["--stage", "train"])
    assert args.stage == "train"


def test_get_time_limit_fallback():
    """_get_time_limit falls back to experiment_minutes if per-stage not set."""
    from run_loop import _get_time_limit

    mock_spec = MagicMock()
    mock_spec.limits.experiment_minutes = 15
    mock_spec.limits.stage1_experiment_minutes = None
    mock_spec.limits.stage2_experiment_minutes = None

    assert _get_time_limit(mock_spec, "prompt") == 15
    assert _get_time_limit(mock_spec, "train") == 15
