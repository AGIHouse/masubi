# Issue 012: Test name test_moe_config_validates_param_budget is misleading

## Severity
Low

## Category
Test Gap

## Description
In `tests/test_schema_validation.py:218-229`, the test `test_moe_config_validates_param_budget` actually tests the `top_k` cap (setting `top_k=5` when `max_top_k=4`), not the parameter budget. The test name suggests it validates total parameter count against `max_params_m`, but it validates `top_k` against `max_top_k`.

There is no actual test for parameter budget validation at the schema level. The `check_param_budget()` function in `student.py` is tested in `test_moe_model.py:test_moe_param_budget_enforced`, but only for a model that is within budget (it does not test the rejection case).

## Evidence
- File: `tests/test_schema_validation.py:218-229` -- test name says "param_budget" but tests top_k
- File: `tests/test_moe_model.py:137-152` -- `test_moe_param_budget_enforced` only tests the passing case

## Suggested Fix
1. Rename `test_moe_config_validates_param_budget` to `test_moe_config_validates_top_k_cap` in `test_schema_validation.py`
2. Add a proper param budget test that creates a model exceeding `max_params_m` and verifies `check_param_budget()` raises:
```python
def test_moe_param_budget_rejection(spec):
    """check_param_budget raises ValueError for model exceeding max_params_m."""
    from autotrust.student import MoEStudent, check_param_budget
    # Create a deliberately oversized model
    big_config = StudentConfig(
        hidden_size=2048, num_layers=24, vocab_size=100000,
        max_seq_len=1024, num_axes=10, num_reason_tags=20,
    )
    moe_config = MoEConfig(num_experts=16, top_k=4, moe_layers=list(range(24)))
    model = MoEStudent.from_config(big_config, moe_config)
    with pytest.raises(ValueError, match="exceeds budget"):
        check_param_budget(model, spec)
```

## Affected Files
- `tests/test_schema_validation.py`
- `tests/test_moe_model.py`
