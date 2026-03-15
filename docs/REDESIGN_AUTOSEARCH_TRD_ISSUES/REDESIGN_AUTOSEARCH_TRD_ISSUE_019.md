# Issue 019: Existing issues 001, 002, 004, 005, 006, 008 are stale (resolved)

## Severity
Low

## Category
Quality

## Description
Six previously filed issues describe problems that have since been resolved in the codebase. The Review Summary in the TRD still references these as open, which creates a misleading picture of project status.

Specifically:
- **Issue 001** ("Stage 2 execution path not implemented"): `_run_stage2_iteration()` is now fully implemented in `run_loop.py:362-465` with subprocess execution, checkpoint discovery, and student model scoring.
- **Issue 002** ("train.py archive and Stage 2 template rewrite not implemented"): Both `_archive_train_py()` (line 273) and `_write_stage2_train_py_template()` (line 287) are implemented and called from `_auto_transition()`.
- **Issue 004** ("MoE load balance loss is constant"): The load balance loss now uses proper Switch Transformer style computation with hard top-k assignments for f_i and soft probabilities for P_i (lines 365-388).
- **Issue 005** ("validate_moe_config duplicated"): `validate_moe_config` is now imported from `autotrust.schemas` only (line 14 of student.py).
- **Issue 006** ("MoE layers additive instead of replacement"): `TransformerMoELayer` (lines 391-436) implements proper replacement with its own self-attention + MoE FFN structure.
- **Issue 008** ("Stage 2 agent prompt not implemented"): `_build_agent_prompt()` now includes Stage 2 context (lines 214-232).

## Evidence
- File: `run_loop.py:362-465` -- `_run_stage2_iteration()` exists
- File: `run_loop.py:273-358` -- `_archive_train_py()` and `_write_stage2_train_py_template()` exist
- File: `autotrust/student.py:365-388` -- load balance loss is computed dynamically
- File: `autotrust/student.py:14` -- `from autotrust.schemas import ... validate_moe_config`
- File: `autotrust/student.py:391-436` -- `TransformerMoELayer` replaces FFN
- File: `run_loop.py:214-232` -- Stage 2 agent prompt context

## Suggested Fix
Mark issues 001, 002, 004, 005, 006, and 008 as resolved. Update the Review Summary in the TRD to reflect the current state.

## Affected Files
- `docs/REDESIGN_AUTOSEARCH_TRD_ISSUES/REDESIGN_AUTOSEARCH_TRD_ISSUE_001.md`
- `docs/REDESIGN_AUTOSEARCH_TRD_ISSUES/REDESIGN_AUTOSEARCH_TRD_ISSUE_002.md`
- `docs/REDESIGN_AUTOSEARCH_TRD_ISSUES/REDESIGN_AUTOSEARCH_TRD_ISSUE_004.md`
- `docs/REDESIGN_AUTOSEARCH_TRD_ISSUES/REDESIGN_AUTOSEARCH_TRD_ISSUE_005.md`
- `docs/REDESIGN_AUTOSEARCH_TRD_ISSUES/REDESIGN_AUTOSEARCH_TRD_ISSUE_006.md`
- `docs/REDESIGN_AUTOSEARCH_TRD_ISSUES/REDESIGN_AUTOSEARCH_TRD_ISSUE_008.md`
- `docs/REDESIGN_AUTOSEARCH_TRD.md` (Review Summary section)

## Status: Fixed
All six stale issues (001, 002, 004, 005, 006, 008) marked as "## Status: Fixed" with explanations.
