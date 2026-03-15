# Issue 018: Trust loss uses MSE instead of KL divergence as specified in TRD

## Severity
Medium

## Category
Quality

## Description
TRD TASK_003 specifies that `compute_trust_loss()` should use "KL divergence between predicted and teacher soft labels." The implementation in `autotrust/student.py:120-131` uses MSE loss after sigmoid instead:

```python
def compute_trust_loss(logits: Tensor, soft_targets: Tensor) -> Tensor:
    predictions = torch.sigmoid(logits)
    return F.mse_loss(predictions, soft_targets)
```

MSE is a reasonable choice for regression on soft labels, but it differs from the TRD specification. KL divergence is more principled for distillation from teacher soft labels because it directly measures the information loss between the teacher and student distributions. This is standard in knowledge distillation (Hinton et al., 2015).

The choice of loss function directly impacts training dynamics and convergence quality for the student model.

## Evidence
- File: `autotrust/student.py:120-131` -- uses `F.mse_loss(predictions, soft_targets)`
- TRD TASK_003 Implementation Step 4: `compute_trust_loss(logits, soft_targets) -> Tensor: """KL divergence between predicted and teacher soft labels."""`

## Suggested Fix
Replace MSE with KL divergence:
```python
def compute_trust_loss(logits: Tensor, soft_targets: Tensor) -> Tensor:
    """KL divergence between predicted and teacher soft labels."""
    predictions = F.log_softmax(logits, dim=-1)
    targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return F.kl_div(predictions, targets, reduction="batchmean")
```

Alternatively, if MSE is intentional (e.g., because trust scores are independent axes, not a probability distribution), document the deviation from the TRD and update the docstring.

## Affected Files
- `autotrust/student.py`
- `tests/test_student_model.py` (update `test_trust_loss_soft_labels` if loss function changes)

## Status: Fixed
Documented the MSE vs KL divergence design decision in the docstring. MSE is the correct choice because trust axes are independent scores in [0,1], not a probability distribution.
