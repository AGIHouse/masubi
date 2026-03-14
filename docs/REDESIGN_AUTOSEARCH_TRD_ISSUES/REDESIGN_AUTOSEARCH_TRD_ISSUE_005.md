# Issue 005: validate_moe_config() duplicated between schemas.py and student.py

## Severity
Medium

## Category
DRY Violation

## Description
`validate_moe_config()` is defined identically in two places:
- `autotrust/schemas.py:180` -- typed with `spec: Spec`
- `autotrust/student.py:528` -- typed with `spec` (untyped)

Both functions have the same logic: check `num_experts` against `max_experts` and `top_k` against `max_top_k`. Tests import from both locations (`test_schema_validation.py` from schemas, `test_moe_model.py` from student).

TASK_012 Review Notes claim "DRY: no duplicate spec loading, no duplicate metric computation outside eval.py" but missed this duplication.

## Evidence
- File: `autotrust/schemas.py:180-194` -- first definition
- File: `autotrust/student.py:528-543` -- identical second definition
- File: `tests/test_schema_validation.py:207,220` -- imports from schemas
- File: `tests/test_moe_model.py:115,127` -- imports from student

## Suggested Fix
1. Keep `validate_moe_config()` only in `schemas.py` (authoritative location for validation)
2. Remove `validate_moe_config()` from `student.py`
3. Update `tests/test_moe_model.py` to import from `autotrust.schemas` instead of `autotrust.student`
4. If `student.py` needs the function internally, import it: `from autotrust.schemas import validate_moe_config`

## Affected Files
- `autotrust/student.py`
- `autotrust/schemas.py`
- `tests/test_moe_model.py`
