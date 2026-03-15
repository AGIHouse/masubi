# Issue 003: relabel_training_data() not implemented in freeze.py

## Severity
High

## Category
Omission

## Description
The TRD section 4.2 step 4 requires: "Label all training data using the frozen teacher prompts -> `synth_data/*.jsonl` with soft trust vectors." The function `relabel_training_data(artifacts, spec)` is specified in TASK_004 but was explicitly deferred with the note: "relabel_training_data() deferred -- requires ScoringProvider integration that depends on Wave 4."

This function is critical for Stage 2 because the student model trains on soft teacher labels. Without re-labeled training data, there are no training targets for the student model. The Stage 2 pipeline cannot function without this.

## Evidence
- File: `autotrust/freeze.py` -- no `relabel_training_data()` function exists
- TRD Section 4.2 Step 4: "Label all training data using the frozen teacher prompts"
- TASK_004 Step 6: `relabel_training_data()` -- specified but deferred
- TASK_004 Review Notes: "relabel_training_data() deferred"

## Suggested Fix
Implement `relabel_training_data()` in `freeze.py`:
```python
def relabel_training_data(artifacts: TeacherArtifacts, spec: Spec) -> Path:
    """Re-label training data using frozen teacher prompts.

    Loads existing synth_data/train.jsonl, scores each chain using the
    frozen teacher prompts via ScoringProvider, and writes updated JSONL
    with soft trust vectors as training targets.
    """
    from autotrust.providers import get_provider

    scorer_provider = get_provider("scorer", spec)
    # Load prompt pack from frozen artifacts
    prompt_pack = yaml.safe_load(artifacts.prompt_pack_path.read_text())

    # Read existing training data
    input_path = artifacts.synth_data_dir / "train.jsonl"
    output_path = artifacts.synth_data_dir / "train_labeled.jsonl"

    # Score each chain and write soft labels
    ...
    return output_path
```

Add corresponding tests in `tests/test_freeze.py`.

## Affected Files
- `autotrust/freeze.py`
- `tests/test_freeze.py`
