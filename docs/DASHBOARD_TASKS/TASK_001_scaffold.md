# Task 001: Dashboard Scaffold & Dependencies

## Context
The Gradio Dashboard is an optional add-on for monitoring the autoresearch loop. It requires its own package directory (`autotrust/dashboard/`) and optional dependencies (gradio, plotly, pandas). The dashboard must not affect the core autoresearch loop -- `gradio` is an optional dependency, and the existing code must continue to work without it. See GRADIO_DASHBOARD_PRD.md sections 5.1 (Architecture) and 5.9 (Dependencies).

## Goal
Create the dashboard package skeleton and add optional dependencies so that `uv sync --extra dashboard` installs Gradio/Plotly/Pandas and `uv run python -c "from autotrust.dashboard import *"` succeeds.

## Research First
- [ ] Read GRADIO_DASHBOARD_PRD.md section 5.1 (Architecture) for the file layout
- [ ] Read GRADIO_DASHBOARD_PRD.md section 5.9 (Dependencies) for required packages
- [ ] Read `pyproject.toml` to understand current dependency structure
- [ ] Verify `autotrust/__init__.py` exists

## TDD: Tests First (Red)
No unit tests for this task (it is pure configuration/scaffold). Verification is that imports succeed.

## Implementation
- [ ] Step 1: Add `[project.optional-dependencies]` entry for dashboard in `pyproject.toml`:
  ```toml
  [project.optional-dependencies]
  dev = ["pytest", "ruff"]
  dashboard = [
      "gradio>=5.0",
      "plotly>=5.0",
      "pandas>=2.0",
  ]
  ```
- [ ] Step 2: Create the dashboard package skeleton:
  - `autotrust/dashboard/__init__.py` (empty, or with `__all__` listing submodules)
  - `autotrust/dashboard/run_manager.py` (empty placeholder with docstring)
  - `autotrust/dashboard/data_loader.py` (empty placeholder with docstring)
  - `autotrust/dashboard/git_history.py` (empty placeholder with docstring)
  - `autotrust/dashboard/charts.py` (empty placeholder with docstring)
  - `autotrust/dashboard/log_formatter.py` (empty placeholder with docstring)
- [ ] Step 3: Create `dashboard.py` at project root (empty placeholder with docstring -- Gradio Blocks app entry point)
- [ ] Step 4: Run `uv sync --extra dashboard` to verify installation succeeds
- [ ] Step 5: Run `uv run python -c "from autotrust.dashboard import data_loader, git_history, charts, log_formatter, run_manager"` to verify imports

## TDD: Tests Pass (Green)
- [ ] `uv sync --extra dashboard` completes without error
- [ ] `uv run python -c "from autotrust.dashboard import data_loader"` succeeds
- [ ] All 103 existing tests still pass

## Acceptance Criteria
- [ ] `pyproject.toml` has `dashboard` optional dependency group with gradio, plotly, pandas
- [ ] `autotrust/dashboard/__init__.py` exists
- [ ] All 5 submodule placeholder files exist under `autotrust/dashboard/`
- [ ] `dashboard.py` exists at project root
- [ ] `uv sync --extra dashboard` succeeds
- [ ] All existing tests pass unchanged

## Execution
- **Agent Type**: infra-sre-architect
- **Wave**: 1
- **Complexity**: Low
