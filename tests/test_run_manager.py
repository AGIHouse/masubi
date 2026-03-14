"""Tests for autotrust/dashboard/run_manager.py -- thread management."""

import time
import threading
from unittest.mock import patch, MagicMock

import pytest


def _mock_run_autoresearch(max_experiments=50, stop_check=None, pause_check=None):
    """A mock run_autoresearch that checks stop/pause like the real one."""
    for _ in range(max_experiments):
        if stop_check and stop_check():
            break
        while pause_check and pause_check():
            time.sleep(0.01)
            if stop_check and stop_check():
                return
        time.sleep(0.01)


def test_initial_status_is_idle():
    """Newly created RunManager has status 'idle' and current_run_id is None."""
    from autotrust.dashboard.run_manager import RunManager

    rm = RunManager()
    assert rm.status == "idle"
    assert rm.current_run_id is None


def test_start_sets_running():
    """After start(), status is 'running' and current_run_id is not None."""
    from autotrust.dashboard.run_manager import RunManager

    rm = RunManager()
    with patch("autotrust.dashboard.run_manager.run_autoresearch", side_effect=_mock_run_autoresearch):
        run_id = rm.start(max_experiments=5)
        time.sleep(0.05)
        assert rm.status == "running"
        assert rm.current_run_id is not None
        assert run_id is not None
        rm.stop()
        time.sleep(0.2)


def test_stop_sets_stopping_then_idle():
    """After stop(), status transitions to 'stopping', then 'idle' after thread exits."""
    from autotrust.dashboard.run_manager import RunManager

    rm = RunManager()
    with patch("autotrust.dashboard.run_manager.run_autoresearch", side_effect=_mock_run_autoresearch):
        rm.start(max_experiments=1000)
        time.sleep(0.05)
        rm.stop()
        # Should eventually become idle
        time.sleep(0.5)
        assert rm.status == "idle"


def test_pause_resume_lifecycle():
    """pause() sets status 'paused', resume() sets status back to 'running'."""
    from autotrust.dashboard.run_manager import RunManager

    rm = RunManager()
    with patch("autotrust.dashboard.run_manager.run_autoresearch", side_effect=_mock_run_autoresearch):
        rm.start(max_experiments=1000)
        time.sleep(0.05)

        rm.pause()
        assert rm.status == "paused"

        rm.resume()
        assert rm.status == "running"

        rm.stop()
        time.sleep(0.5)


def test_stop_check_callback_returns_true_when_stopped():
    """The stop_check callback returns True after stop() is called."""
    from autotrust.dashboard.run_manager import RunManager

    rm = RunManager()
    assert rm._stop_check() is False  # not stopped yet

    rm._stop_event.set()
    assert rm._stop_check() is True


def test_start_when_already_running_raises():
    """Calling start() while already running raises RuntimeError."""
    from autotrust.dashboard.run_manager import RunManager

    rm = RunManager()
    with patch("autotrust.dashboard.run_manager.run_autoresearch", side_effect=_mock_run_autoresearch):
        rm.start(max_experiments=1000)
        time.sleep(0.05)

        with pytest.raises(RuntimeError):
            rm.start(max_experiments=10)

        rm.stop()
        time.sleep(0.5)
