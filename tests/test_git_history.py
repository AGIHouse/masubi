"""Tests for autotrust/dashboard/git_history.py -- git diff & log parser."""

import subprocess
from unittest.mock import patch, MagicMock


def test_get_train_py_log_parses_output():
    """Mock subprocess to return known git log output, verify returns correct dicts."""
    from autotrust.dashboard.git_history import get_train_py_log

    mock_output = (
        "abc1234|||experiment 5: keep|||2024-03-14 14:35:00 -0500\n"
        "def5678|||experiment 3: keep|||2024-03-14 14:23:00 -0500\n"
        "890abcd|||experiment 1: keep|||2024-03-14 14:20:00 -0500"
    )
    mock_result = MagicMock()
    mock_result.stdout = mock_output
    mock_result.returncode = 0

    with patch("autotrust.dashboard.git_history.subprocess.run", return_value=mock_result):
        commits = get_train_py_log()

    assert len(commits) == 3
    assert commits[0]["hash"] == "abc1234"
    assert commits[0]["message"] == "experiment 5: keep"
    assert "date" in commits[0]


def test_get_train_py_log_empty_repo():
    """Mock subprocess returning empty output, returns empty list."""
    from autotrust.dashboard.git_history import get_train_py_log

    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.returncode = 0

    with patch("autotrust.dashboard.git_history.subprocess.run", return_value=mock_result):
        commits = get_train_py_log()

    assert commits == []


def test_get_diff_returns_unified_diff():
    """Mock subprocess to return known diff output, verify returns diff string."""
    from autotrust.dashboard.git_history import get_diff

    diff_text = "--- a/train.py\n+++ b/train.py\n@@ -1,3 +1,4 @@\n+import torch\n"
    mock_result = MagicMock()
    mock_result.stdout = diff_text
    mock_result.returncode = 0

    with patch("autotrust.dashboard.git_history.subprocess.run", return_value=mock_result):
        result = get_diff("abc1234", "def5678")

    assert "import torch" in result
    assert "--- a/train.py" in result


def test_get_file_at_commit_returns_content():
    """Mock subprocess to return file content, verify returns string."""
    from autotrust.dashboard.git_history import get_file_at_commit

    file_content = "import os\n\ndef main():\n    pass\n"
    mock_result = MagicMock()
    mock_result.stdout = file_content
    mock_result.returncode = 0

    with patch("autotrust.dashboard.git_history.subprocess.run", return_value=mock_result):
        result = get_file_at_commit("abc1234")

    assert "import os" in result
    assert "def main():" in result


def test_get_diff_nonzero_returncode_returns_empty():
    """If git diff returns non-zero exit code, return empty string."""
    from autotrust.dashboard.git_history import get_diff

    mock_result = MagicMock()
    mock_result.stdout = "fatal: bad revision 'badref'"
    mock_result.stderr = "fatal: bad revision 'badref'"
    mock_result.returncode = 128

    with patch("autotrust.dashboard.git_history.subprocess.run", return_value=mock_result):
        result = get_diff("abc1234", "def5678")

    assert result == ""


def test_get_file_at_commit_nonzero_returncode_returns_empty():
    """If git show returns non-zero exit code, return empty string."""
    from autotrust.dashboard.git_history import get_file_at_commit

    mock_result = MagicMock()
    mock_result.stdout = "fatal: bad object abc1234"
    mock_result.stderr = "fatal: bad object abc1234"
    mock_result.returncode = 128

    with patch("autotrust.dashboard.git_history.subprocess.run", return_value=mock_result):
        result = get_file_at_commit("abc1234")

    assert result == ""


def test_subprocess_timeout_handled():
    """Mock subprocess to raise TimeoutExpired, verify returns empty/default gracefully."""
    from autotrust.dashboard.git_history import get_train_py_log, get_diff, get_file_at_commit

    with patch(
        "autotrust.dashboard.git_history.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="git", timeout=10),
    ):
        assert get_train_py_log() == []
        assert get_diff("abc", "def") == ""
        assert get_file_at_commit("abc") == ""
