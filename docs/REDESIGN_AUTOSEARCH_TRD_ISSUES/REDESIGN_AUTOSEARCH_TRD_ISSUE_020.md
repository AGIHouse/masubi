# Issue 020: LocalInference.should_escalate uses 0.9 threshold heuristic instead of model's escalate flag

## Severity
Medium

## Category
Bug

## Description
The `LocalInference.should_escalate()` method (instance method on the class, at `inference.py:112-129`) receives a `ScorerOutput` which does not carry the raw `escalate` flag from the student model. As a workaround, it uses a heuristic: `any(v >= 0.9 for v in output.trust_vector.values())`. This means escalation decisions are based on an arbitrary 0.9 threshold on trust scores rather than the model's trained escalation head output.

Meanwhile, the standalone `should_escalate()` function at module level (lines 57-69) correctly uses `student_output.escalate`. The two functions with the same name but different semantics create confusion.

The TRD section 4.8 specifies: `should_escalate(scorer_output, spec) -> bool` -- "checks escalate head output". The instance method does not check the escalate head.

## Evidence
- File: `autotrust/inference.py:129` -- `return any(v >= 0.9 for v in output.trust_vector.values())`
- File: `autotrust/inference.py:57-69` -- standalone `should_escalate(student_output, spec)` correctly uses `student_output.escalate`
- File: `autotrust/inference.py:127-128` -- comment: "since ScorerOutput doesn't carry the raw escalate flag"
- TRD Section 4.8: `should_escalate(scorer_output, spec) -> bool` -- checks escalate head output

## Suggested Fix
Propagate the escalate flag through the pipeline. Options:

1. Add an `escalate` field to `ScorerOutput` (backward-compatible with `Optional[bool] = None`)
2. Have `student_output_to_scorer_output()` preserve the escalate flag in a way the method can access
3. Store the last `StudentOutput` on `LocalInference` and check it in `should_escalate()`

Option 3 is simplest:
```python
def score_text(self, text, axis_names, reason_tag_names, threshold=0.5):
    ...
    self._last_student_output = student_output
    return student_output_to_scorer_output(student_output)

def should_escalate(self, output, spec):
    if self._last_student_output is not None:
        return should_escalate(self._last_student_output, spec)
    return False
```

## Affected Files
- `autotrust/inference.py`
- `tests/test_inference.py`
