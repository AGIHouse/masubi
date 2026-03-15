# Issue 007: LocalInference API does not match TRD specification

## Severity
Medium

## Category
Omission

## Description
The TRD section 4.8 defines `LocalInference` with methods that accept `EmailChain` objects:
- `score(email_chain) -> ScorerOutput`
- `should_escalate(scorer_output, spec) -> bool`

The implementation instead has:
- `score_text(text, axis_names, reason_tag_names, threshold)` -- takes raw text string
- `should_escalate` is a module-level function, not a method on LocalInference

The TRD also says LocalInference should "integrate with existing JudgeProvider for fallback" and that `should_escalate` should "check escalate head output." The implementation delegates axis_names and reason_tag_names to the caller rather than loading them from spec.

This makes the API less ergonomic and requires callers to manually extract text from EmailChain objects and know the axis/tag names.

## Evidence
- File: `autotrust/inference.py:109-135` -- `score_text()` takes `text: str` not `EmailChain`
- File: `autotrust/inference.py:57-69` -- `should_escalate` is module-level, not a method
- TRD Section 4.8: `score(email_chain) -> ScorerOutput` -- takes EmailChain

## Suggested Fix
1. Add `score(chain: EmailChain) -> ScorerOutput` method that:
   - Concatenates emails from the chain into text
   - Gets axis_names from `get_spec()`
   - Calls `score_text()` internally
2. Move `should_escalate()` to be a method on `LocalInference`
3. Keep `score_text()` as a lower-level alternative for raw text input

```python
def score(self, chain: EmailChain) -> ScorerOutput:
    spec = get_spec()
    text = "\n".join(f"{e.subject}\n{e.body}" for e in chain.emails)
    axis_names = [a.name for a in spec.trust_axes]
    return self.score_text(text, axis_names, self._default_reason_tags())
```

## Affected Files
- `autotrust/inference.py`
- `tests/test_inference.py`
