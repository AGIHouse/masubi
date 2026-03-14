# Issue 022: Stage 2 test coverage only verifies helpers, not actual execution path

## Severity
High

## Category
Test Gap

## Description
The tests in `test_stage_transition.py` verify Stage 2 helper functions (CLI parsing, time limits, auto-transition trigger, subprocess call mocking) but do not test the actual Stage 2 execution path end-to-end. Specifically:

1. `test_stage2_runs_subprocess` mocks `_score_with_student_model` to return `None`, so the test verifies subprocess was called but not that the full pipeline (agent edit -> subprocess -> checkpoint -> scoring -> three-gate eval -> keep/discard) works together.
2. No test verifies that `run_autoresearch(stage="train")` actually runs Stage 2 iterations.
3. No test verifies that after auto-transition, the stage variable changes and subsequent iterations use the Stage 2 code path.
4. No test verifies that Stage 2 scoring results flow through the three-gate evaluation.

The main loop in `run_autoresearch()` is over 280 lines with complex branching, and the Stage 2 branch (lines 657-673) is only exercised through mocked helper tests.

## Evidence
- File: `tests/test_stage_transition.py:83-109` -- `test_stage2_runs_subprocess` mocks scoring to None
- File: `run_loop.py:657-673` -- Stage 2 branch in main loop
- File: `run_loop.py:800-804` -- auto-transition in main loop
- TRD Section 5: Test table lists `test_stage_transition.py` for "Handoff trigger, artifact freezing, train.py rewrite"

## Suggested Fix
Add integration-level tests:
1. Test that `run_autoresearch(stage="train", max_experiments=1)` with mocked agent and checkpoint runs the complete Stage 2 pipeline including three-gate evaluation.
2. Test that auto-transition mid-loop changes the execution path from Stage 1 to Stage 2.
3. Test that Stage 2 results are logged correctly in `all_results`.

## Affected Files
- `tests/test_stage_transition.py`
