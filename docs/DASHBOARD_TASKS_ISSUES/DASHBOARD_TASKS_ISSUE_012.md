# Issue 012: git_history.get_diff() and get_file_at_commit() don't check returncode

## Severity
Medium

## Category
Bug

## Description
`get_diff()` and `get_file_at_commit()` call `subprocess.run()` and return `result.stdout` without checking `result.returncode`. If git returns a non-zero exit code (e.g., the ref doesn't exist, the file doesn't exist at that commit, or the ref is ambiguous), the function returns stderr mixed into stdout, or an empty string, without any indication of failure.

For example, if a user selects two invalid commits, the diff display might show a git error message ("fatal: bad revision 'abc1234'") as if it were a valid diff.

## Evidence
- File: `autotrust/dashboard/git_history.py:77-83` -- `get_diff()` returns `result.stdout` without checking returncode
- File: `autotrust/dashboard/git_history.py:103-110` -- `get_file_at_commit()` returns `result.stdout` without checking returncode
- PRD Requirement: TASK_003 Acceptance Criteria -- "Graceful fallback on subprocess errors (return empty, log warning)"

## Suggested Fix
Check returncode and return empty string with a warning if non-zero:
```python
if result.returncode != 0:
    logger.warning("git diff failed (rc=%d): %s", result.returncode, result.stderr)
    return ""
return result.stdout
```

## Affected Files
- `autotrust/dashboard/git_history.py`
