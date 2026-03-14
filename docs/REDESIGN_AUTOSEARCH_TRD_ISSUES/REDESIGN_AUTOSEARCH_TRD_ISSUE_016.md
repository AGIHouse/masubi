# Issue 016: relabel_training_data() full path calls wrong ScoringProvider API

## Severity
High

## Category
Bug

## Description
In `autotrust/freeze.py`, the `relabel_training_data()` function's full relabeling path (lines 376-408) calls `scorer_provider.score(text, spec)` with two arguments. However, the `ScoringProvider` protocol's `score()` method takes only a single argument: `score(prompt: str) -> str`. This means the full relabeling path would fail with a `TypeError` at runtime.

The fallback path (lines 352-374, triggered when `get_provider()` fails) works correctly because it uses existing labels as soft targets without calling the scorer.

## Evidence
- File: `autotrust/freeze.py:391` -- `scorer_output = scorer_provider.score(text, spec)` -- passes 2 args
- File: `autotrust/providers/__init__.py` or `autotrust/providers/hyperbolic.py` -- `ScoringProvider.score(prompt: str) -> str` takes 1 arg
- TRD Section 4.2 Step 4: "Label all training data using the frozen teacher prompts"
- The function also treats the return value as a `ScorerOutput` object (accessing `.trust_vector`), but `score()` returns a raw string

## Suggested Fix
The relabeling should construct a prompt using the frozen teacher artifacts and parse the raw response:
```python
from train_stage1_archive import EmailTrustScorer

scorer = EmailTrustScorer(provider=scorer_provider, spec=spec)
for record in records:
    chain = EmailChain.model_validate(record)
    scorer_output = scorer.score_chain(chain)
    labeled = dict(record)
    labeled["soft_targets"] = scorer_output.trust_vector
    labeled_records.append(labeled)
```

Or, if `EmailTrustScorer` is unavailable after Stage 2 transition, build prompts manually from the frozen `prompt_pack.yaml`.

## Affected Files
- `autotrust/freeze.py`
