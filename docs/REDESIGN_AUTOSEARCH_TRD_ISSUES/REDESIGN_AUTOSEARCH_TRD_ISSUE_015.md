# Issue 015: train.py overwritten with Stage 2 template breaks 5 existing tests

## Severity
Critical

## Category
Bug

## Description
The working directory contains a `train.py` that has been replaced with the Stage 2 PyTorch training template (the output of `_write_stage2_train_py_template()`). This removes the `EmailTrustScorer` class that `tests/test_train.py` imports. As a result, all 5 tests in `test_train.py` fail with `ImportError: cannot import name 'EmailTrustScorer' from 'train'`.

The TRD section 5 states: "All 20 test files in `tests/` must continue to pass. The redesign is additive." This is a regression.

The original Stage 1 code is preserved in `train_stage1_archive.py` (identical content), confirming that `_archive_train_py()` ran but `train.py` was not restored afterward.

## Evidence
- File: `train.py` -- git diff shows the entire `EmailTrustScorer` class was replaced with Stage 2 template
- File: `train_stage1_archive.py` -- contains the original `EmailTrustScorer` code
- File: `tests/test_train.py:62` -- `from train import EmailTrustScorer` fails with ImportError
- Test output: 5 FAILED (`test_scorer_returns_scorer_output`, `test_scorer_output_has_trust_vector`, `test_scorer_output_has_explanation`, `test_scorer_batch`, `test_scorer_reasons_are_strings`)
- TRD Section 5: "All 20 test files in `tests/` must continue to pass"

## Suggested Fix
Two options:

**Option A (immediate):** Restore `train.py` to its Stage 1 content:
```bash
cp train_stage1_archive.py train.py
```

**Option B (structural):** Update `tests/test_train.py` to import from `train_stage1_archive` when `train.py` no longer contains `EmailTrustScorer`, or make the tests conditional on the current stage:
```python
try:
    from train import EmailTrustScorer
except ImportError:
    from train_stage1_archive import EmailTrustScorer
```

Option A is preferred because `train.py` should remain as the Stage 1 scorer until auto-transition actually fires during a real run.

## Affected Files
- `train.py`
- `train_stage1_archive.py`
- `tests/test_train.py`

## Status: Fixed
Created `starting_train.py` as the canonical Stage 1 template. `train.py` is now the ephemeral working copy that the agent edits and that gets overwritten at Stage 2 transition. Tests import from `starting_train`. `run_loop.py` copies `starting_train.py` -> `train.py` at the start of a run. `train_stage1_archive.py` is redundant (identical to `starting_train.py`).
