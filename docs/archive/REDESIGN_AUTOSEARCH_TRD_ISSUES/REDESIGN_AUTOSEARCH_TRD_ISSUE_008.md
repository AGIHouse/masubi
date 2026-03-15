# Issue 008: Stage 2 agent prompt not implemented

## Severity
High

## Category
Omission

## Description
The TRD section 4.5 (step 7) requires `_build_agent_prompt()` to include stage-specific instructions:
- Stage 1: current prompt (prompt optimization)
- Stage 2: include `program.md` Stage 2 instructions, model architecture constraints from spec.yaml

The current `_build_agent_prompt()` (run_loop.py:175-201) always builds a Stage 1 prompt. It includes the LoRA nudge at 3+ no-improvement (line 193), but does not include Stage 2 architecture constraints, training data references, or MoE caps when `stage == "train"`.

Even though `program.md` now has a Stage 2 section (TASK_009), the agent prompt sends the full program.md without directing the agent to focus on Stage 2 instructions. More importantly, the agent prompt does not include:
- The student model config constraints from spec.yaml
- The current training loss/metrics from the last checkpoint
- References to `autotrust/student.py` architecture classes

## Evidence
- File: `run_loop.py:175-201` -- `_build_agent_prompt()` has no stage parameter or stage-conditional logic
- TRD Section 4.5 Step 7: "Update `_build_agent_prompt()` to include stage-specific instructions"
- TASK_008 Step 7: "Stage 2: include program.md Stage 2 instructions, model architecture constraints from spec.yaml"

## Suggested Fix
1. Add `stage` parameter to `_build_agent_prompt()`
2. When `stage == "train"`, replace the LoRA nudge with Stage 2 context:
   - Include `spec.stage2` constraints (max_experts, max_params_m, max_top_k)
   - Include the current checkpoint metrics if available
   - Include `StudentConfig` and `MoEConfig` schema definitions
   - Reference the `autotrust/student.py` API (DenseStudent, MoEStudent)

## Affected Files
- `run_loop.py`
- `tests/test_stage_transition.py`

## Status: Fixed
`_build_agent_prompt()` now includes a `stage` parameter. When `stage == "train"`, it includes Stage 2 context: spec.yaml architecture constraints, available APIs, and output requirements.
