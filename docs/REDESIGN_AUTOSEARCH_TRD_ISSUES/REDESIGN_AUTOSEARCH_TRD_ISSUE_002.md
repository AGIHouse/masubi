# Issue 002: train.py archive and Stage 2 template rewrite not implemented

## Severity
Critical

## Category
Omission

## Description
The TRD section 4.4 states: "At stage handoff, `train.py` is rewritten from scratch. The old prompt-based code is archived." The `_auto_transition()` function in `run_loop.py` only calls `freeze_teacher(spec)` but does not:

1. Archive the current `train.py` (e.g., via git branch or copy to `archive/`)
2. Replace `train.py` with a Stage 2 PyTorch training template that imports from `autotrust.student` and implements a training loop

Without this, even after auto-transition, the system continues reading the Stage 1 prompt-based `train.py` and trying to use it with `EmailTrustScorer`.

TASK_008 Implementation Step 5 lists `_archive_train_py()` and `_write_stage2_train_py_template()` as required functions, but neither was implemented.

## Evidence
- File: `run_loop.py:137-146` -- `_auto_transition()` only calls `freeze_teacher()`, no archive/rewrite
- TRD Section 4.4: "At stage handoff, `train.py` is rewritten from scratch"
- TASK_008 Step 5: `_archive_train_py()` and `_write_stage2_train_py_template()` -- listed but not implemented

## Suggested Fix
1. Implement `_archive_train_py()`:
   ```python
   def _archive_train_py():
       """Archive Stage 1 train.py to a git tag or backup file."""
       subprocess.run(["git", "tag", "stage1-complete"], capture_output=True)
       shutil.copy("train.py", "train_stage1_archive.py")
   ```
2. Implement `_write_stage2_train_py_template()`:
   ```python
   def _write_stage2_train_py_template():
       """Write Stage 2 PyTorch training template for train.py."""
       template = '''"""Stage 2: Student model training script.
       ...imports from autotrust.student, training loop, checkpoint saving...
       """'''
       Path("train.py").write_text(template)
   ```
3. Call both in `_auto_transition()` after `freeze_teacher()`

## Affected Files
- `run_loop.py`
- `tests/test_stage_transition.py`

## Status: Fixed
Both `_archive_train_py()` and `_write_stage2_train_py_template()` are implemented and called from `_auto_transition()`. The canonical template is now `starting_train.py`, and `train.py` is the working copy that gets overwritten during runs.
