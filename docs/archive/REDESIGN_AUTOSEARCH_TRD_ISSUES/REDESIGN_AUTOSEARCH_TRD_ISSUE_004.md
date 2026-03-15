# Issue 004: MoE load balance loss is constant (always ~1.0)

## Severity
Medium

## Category
Bug

## Description
The `_compute_load_balance_loss()` method in `MoEBlock` (student.py:365-376) computes:

```python
expert_usage = router_probs.mean(dim=0)  # (num_experts,)
target = torch.ones_like(expert_usage) / self.num_experts
loss = self.num_experts * (expert_usage * target).sum()
```

This simplifies to: `N * sum(mean(probs, dim=0) * (1/N))` = `sum(mean(probs, dim=0))` = `mean(sum(probs, dim=-1))` = `mean(1.0)` = `1.0`.

Since `router_probs` are softmax outputs that sum to 1 per token, and `expert_usage` is the mean across tokens, `sum(expert_usage) = 1.0` always. Multiplying by `target = 1/N` and summing gives `1/N * sum(expert_usage) = 1/N`. Then multiplying by `N` gives `1.0`.

The loss is always approximately 1.0 regardless of whether routing is balanced or imbalanced. This means the auxiliary loss provides no gradient signal to encourage balanced routing.

The correct Switch Transformer load balance loss is: `L = N * sum(f_i * P_i)` where:
- `f_i` = fraction of tokens routed to expert i (using **hard** top-k assignments)
- `P_i` = mean routing probability for expert i

The current code uses soft probabilities for both `f_i` and `P_i`, which makes them identical (both are `expert_usage`).

## Evidence
- File: `autotrust/student.py:365-376` -- `_compute_load_balance_loss()` implementation
- Test: `test_moe_load_balance_loss` only asserts `aux_loss.item() >= 0` -- does not verify the loss discriminates between balanced and imbalanced routing

## Suggested Fix
Fix `_compute_load_balance_loss` to use hard assignments for `f_i`:
```python
def _compute_load_balance_loss(self, router_probs: Tensor) -> Tensor:
    # f_i: fraction of tokens dispatched to each expert (hard assignment)
    top_k_indices = torch.topk(router_probs, self.top_k, dim=-1).indices
    # Count tokens per expert
    dispatch_count = torch.zeros(self.num_experts, device=router_probs.device)
    for k in range(self.top_k):
        for expert_idx in range(self.num_experts):
            dispatch_count[expert_idx] += (top_k_indices[:, k] == expert_idx).float().sum()
    f = dispatch_count / (router_probs.shape[0] * self.top_k)

    # P_i: mean routing probability per expert (soft)
    P = router_probs.mean(dim=0)

    return self.num_experts * (f * P).sum()
```

Also strengthen the test to verify the loss is larger for imbalanced routing.

## Affected Files
- `autotrust/student.py`
- `tests/test_moe_model.py`

## Status: Fixed
The load balance loss now uses hard top-k assignments for f_i and soft probabilities for P_i (Switch Transformer style), correctly discriminating between balanced and imbalanced routing.
