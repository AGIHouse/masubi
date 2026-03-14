# Issue 001: Stage 2 execution path not implemented in run_loop.py

## Severity
Critical

## Category
Omission

## Description
The TRD section 4.5 requires that Stage 2 mode executes `train.py` as a subprocess (`subprocess.run(["uv", "run", "python", "train.py"])`) instead of importing it as a module. Currently, `run_loop.py` accepts `--stage train` and sets the stage variable, but the main loop (lines 341-543) always runs the Stage 1 code path: importing `EmailTrustScorer` from `train.py` and calling `score_batch()`. There is no `_run_stage2_iteration()` function, no subprocess execution, and no checkpoint-based evaluation.

When the stage is "train", the loop still:
1. Reads `train.py` as source and sends it to the agent for prompt edits
2. Imports `EmailTrustScorer` from `train.py` and calls `score_batch()`
3. Uses LLM API predictions instead of student model checkpoint predictions

This means `--stage train` is functionally identical to `--stage prompt`. The entire Stage 2 training pipeline is a no-op.

## Evidence
- File: `run_loop.py:341-543` -- main loop has no stage-conditional branching for execution
- File: `run_loop.py:419-441` -- always imports EmailTrustScorer and uses LLM scoring regardless of stage
- TRD Section 4.5: "Stage 2 mode: `train.py` is executed as a subprocess instead of imported as a module"
- TRD Section 4.5: "Stage 2 scoring uses the trained student model checkpoint, not the LLM API"
- TASK_008 Implementation Step 4: `_run_stage2_iteration()` -- listed but never implemented

## Suggested Fix
1. Implement `_run_stage2_iteration(experiment_num, spec, run_ctx, ...)` that:
   - Calls the agent to propose `train.py` edits (same as Stage 1)
   - Runs `train.py` as subprocess: `subprocess.run(["uv", "run", "python", "train.py"], timeout=...)`
   - Loads the resulting checkpoint from `runs/<run_id>/checkpoints/`
   - Scores eval chains using the student model via `LocalInference`
   - Runs three-gate evaluation (same policy)
   - Handles git keep/discard
2. Add stage-conditional branching in the main loop:
   ```python
   if stage == "train":
       _run_stage2_iteration(experiment_num, spec, run_ctx, ...)
   else:
       # existing Stage 1 logic
   ```
3. Update `_build_agent_prompt()` to include Stage 2 instructions when `stage == "train"`

## Affected Files
- `run_loop.py`
- `tests/test_stage_transition.py` (needs tests for Stage 2 execution)
