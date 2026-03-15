# Task 001: spec.yaml Stage 2/Production Sections + config.py Models

## Context
The REDESIGN_AUTOSEARCH_TRD requires the system to support Stage 2 (MoE student training) and Stage 3 (production inference). The current `spec.yaml` has no `stage2` or `production` sections, and `config.py` has no models for them. The `limits` section also needs per-stage time limits instead of a single `experiment_minutes`.

## Goal
Add `stage2` and `production` sections to `spec.yaml` and corresponding Pydantic models to `config.py`, maintaining backward compatibility.

## Research First
- [x] Read `/Users/jonathan.hendler/personal/autoresearch-helpful/spec.yaml` (current state)
- [x] Read `/Users/jonathan.hendler/personal/autoresearch-helpful/autotrust/config.py` (current models)
- [x] Read `/Users/jonathan.hendler/personal/autoresearch-helpful/docs/REDESIGN_AUTOSEARCH.md` lines 296-310 (target spec.yaml)
- [x] Verify that existing tests in `test_config.py` pass before making changes
- [x] Check all files that import from `config.py` to ensure no breakage

## TDD: Tests First (Red)
Write tests FIRST -- they should FAIL before implementation.

### Unit Tests
- [x] Test: `test_spec_has_stage2_section` in `tests/test_config.py` -- verify `spec.stage2` exists after loading
- [x] Test: `test_stage2_dense_baseline_first_default` -- verify `stage2.dense_baseline_first` is `True`
- [x] Test: `test_stage2_max_experts_cap` -- verify `stage2.max_experts` is 16
- [x] Test: `test_stage2_max_params_cap` -- verify `stage2.max_params_m` is 200
- [x] Test: `test_stage2_max_top_k_cap` -- verify `stage2.max_top_k` is 4
- [x] Test: `test_stage2_export_formats` -- verify `stage2.export_formats` contains `["pytorch", "gguf"]`
- [x] Test: `test_spec_has_production_section` -- verify `spec.production` exists
- [x] Test: `test_production_judge_fallback_enabled` -- verify `production.judge_fallback_enabled` is `True`
- [x] Test: `test_production_escalate_on_flag` -- verify `production.escalate_on_flag` is `True`
- [x] Test: `test_limits_per_stage_times` -- verify `limits.stage1_experiment_minutes` and `limits.stage2_experiment_minutes` exist

## Implementation
- [x] Step 1: Add `Stage2Config` Pydantic model to `config.py`:
  ```python
  class Stage2Config(BaseModel):
      dense_baseline_first: bool
      max_experts: int
      max_params_m: int
      max_top_k: int
      export_formats: list[str]
  ```
- [x] Step 2: Add `ProductionConfig` Pydantic model to `config.py`:
  ```python
  class ProductionConfig(BaseModel):
      judge_fallback_enabled: bool
      escalate_on_flag: bool
  ```
- [x] Step 3: Update `Limits` model to support per-stage times:
  ```python
  class Limits(BaseModel):
      experiment_minutes: int  # keep for backward compat
      stage1_experiment_minutes: int | None = None
      stage2_experiment_minutes: int | None = None
      max_spend_usd: float
      per_experiment_timeout_minutes: float = 10.0
  ```
- [x] Step 4: Add `stage2` and `production` fields to `Spec` model (both optional for backward compat):
  ```python
  stage2: Stage2Config | None = None
  production: ProductionConfig | None = None
  ```
- [x] Step 5: Add `stage2` and `production` sections to `spec.yaml`
- [x] Step 6: Update `Limits` section in `spec.yaml` with per-stage times
- [x] DRY check: no duplication with existing `Limits`, `JudgeConfig`, or `ExplanationConfig` models

## TDD: Tests Pass (Green)
- [x] All new tests pass
- [x] All existing tests in `test_config.py` still pass
- [x] All existing tests in the full suite still pass

## Acceptance Criteria
- [x] `load_spec()` successfully loads the updated `spec.yaml` with `stage2` and `production`
- [x] `spec.stage2.max_experts` returns 16
- [x] `spec.production.judge_fallback_enabled` returns True
- [x] No existing test is broken
- [x] `spec.yaml` validates (weights still sum to 1.0)

## Review Notes
- Added `Stage2Config` and `ProductionConfig` Pydantic models to `config.py`
- Added `stage2` and `production` optional fields to `Spec` model (backward compatible)
- Added `stage1_experiment_minutes` and `stage2_experiment_minutes` optional fields to `Limits`
- Added `stage2` and `production` sections to `spec.yaml`
- Added per-stage time limits to `spec.yaml` limits section
- 10 new tests added to `test_config.py`, all pass
- All 9 original tests continue to pass
- No DRY violations: Stage2Config/ProductionConfig are distinct from existing models

## Execution
- **Agent Type**: coding subagent
- **Wave**: 1 (parallel with TASK_002)
- **Complexity**: Low
