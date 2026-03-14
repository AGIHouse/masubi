# Issue 006: MoE layers run both standard FFN and MoE FFN (additive instead of replacement)

## Severity
Medium

## Category
Bug

## Description
In `MoEStudent.forward()` (student.py:454-465), for layers designated as MoE layers, the code:

1. Runs the full standard `TransformerEncoderLayer` (which includes self-attention AND feed-forward network)
2. Then additionally runs the `MoEBlock` and adds its output as a residual

```python
if str(i) in self.moe_blocks:
    x = layer(x, src_key_padding_mask=src_key_padding_mask)  # full layer incl FFN
    moe_out, aux_loss = self.moe_blocks[str(i)](x)
    x = x + moe_out  # additive MoE on top
```

This means MoE layers have both a standard FFN (inside the TransformerEncoderLayer) AND a separate MoE FFN. The TRD section 4.3 says "selected layers replaced with MoE blocks" -- "replaced" implies the standard FFN should be removed and substituted with the MoE block, not added on top.

This has two consequences:
- Double the computation and parameters for MoE layers (both FFNs run)
- The MoE routing signal is diluted since the standard FFN already processes the input

The TASK_006 Review Notes say "MoE layers share attention from standard TransformerEncoderLayer, add MoE FFN as residual" -- confirming this is intentional but inconsistent with the TRD's "replaced."

## Evidence
- File: `autotrust/student.py:454-465` -- MoE forward pass
- TRD Section 4.3: "selected layers replaced with MoE blocks"
- TASK_006 Review Notes: "add MoE FFN as residual" (contradicts "replaced")

## Suggested Fix
Replace the standard FFN with MoE in MoE layers. This requires decomposing the `TransformerEncoderLayer` to use its attention sublayer but not its FFN sublayer for MoE-designated layers. Options:

1. **Custom transformer layer**: Create a `TransformerLayerWithMoE` that uses attention from `nn.MultiheadAttention` and MoE for FFN
2. **Post-hoc replacement**: After creating the standard layer, replace its `linear1`/`linear2` weights with the MoE block
3. **Accept the additive design**: If intentional for gradient flow, document it clearly and update TRD

## Affected Files
- `autotrust/student.py`
- `tests/test_moe_model.py`
