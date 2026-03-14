# Issue 011: torch.load with weights_only=False is a security risk

## Severity
Medium

## Category
Security

## Description
Both `load_pytorch()` and `list_checkpoints()` in `export.py` use `torch.load(path, weights_only=False)`. PyTorch's documentation explicitly warns that `weights_only=False` allows arbitrary code execution via pickle deserialization. An attacker who can place a malicious `.pt` file in the checkpoints directory could execute arbitrary code when the checkpoint is loaded.

While this is an internal research tool, the TRD mentions production inference (Stage 3) where checkpoints would be loaded in a deployment context. The GGUF export also loads checkpoints via this path.

## Evidence
- File: `autotrust/export.py:72` -- `torch.load(checkpoint_path, weights_only=False)` in `load_pytorch()`
- File: `autotrust/export.py:144` -- `torch.load(pt_file, weights_only=False)` in `list_checkpoints()`
- PyTorch docs: "Only load data you trust... `weights_only=True` is recommended"

## Suggested Fix
1. Use `torch.load(path, weights_only=True)` and explicitly register safe classes
2. Or add a signature verification step before loading
3. At minimum, add a warning comment documenting the security implication

For `weights_only=True`, the checkpoint structure (state_dict, config dict, meta dict) should be safe to load without pickle since it's all basic types and tensors.

```python
checkpoint = torch.load(checkpoint_path, weights_only=True)
```

If custom classes need to be unpickled, use `torch.serialization.add_safe_globals()`.

## Affected Files
- `autotrust/export.py`
