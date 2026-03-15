# Issue 017: StudentOutput does not use shared validate_trust_vector()

## Severity
Medium

## Category
Omission

## Description
TASK_002 acceptance criteria state: "StudentOutput trust vector validation uses shared validate_trust_vector() function." However, `StudentOutput` in `schemas.py` is a plain `BaseModel` with no model validator for its `trust_vector` field. Unlike `ScorerOutput`, which has a `@model_validator` that calls `validate_trust_vector()` when the spec singleton is loaded, `StudentOutput` performs no validation.

This means a `StudentOutput` can be constructed with arbitrary keys in `trust_vector` (e.g., `{"fake_axis": 0.5}`) without any error. While this may be intentional for flexibility during model development, it violates the stated acceptance criteria and could allow invalid data to flow through the pipeline undetected.

## Evidence
- File: `autotrust/schemas.py:158-162` -- `StudentOutput` has no `@model_validator`
- File: `autotrust/schemas.py:80-90` -- `ScorerOutput` has `_validate_trust_vector` model validator
- TASK_002 Acceptance Criteria: "StudentOutput trust vector validation uses shared validate_trust_vector() function"

## Suggested Fix
Add a model validator to `StudentOutput` that mirrors `ScorerOutput`:
```python
class StudentOutput(BaseModel):
    trust_vector: dict[str, float]
    reason_tags: list[str]
    escalate: bool

    @model_validator(mode="after")
    def _validate_trust_vector(self) -> StudentOutput:
        from autotrust.config import _spec
        if _spec is not None:
            validate_trust_vector(self.trust_vector, _spec)
        return self
```

Note: This may require updating tests that construct `StudentOutput` with partial trust vectors (e.g., `test_escalation_decision_true` uses `{"phish": 0.9}` only).

## Affected Files
- `autotrust/schemas.py`
- `tests/test_inference.py` (may need full trust vectors in test fixtures)

## Status: Fixed
Added `@model_validator` to `StudentOutput` mirroring `ScorerOutput` pattern. Updated test fixtures in `test_inference.py` and `test_schema_validation.py` to use full trust vectors.
