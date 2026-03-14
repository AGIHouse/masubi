# Issue 010: freeze.py and export.py missing CLI modules

## Severity
Medium

## Category
Omission

## Description
The TRD specifies CLI entry points for both freeze and export:
- Section 4.2: "CLI: `uv run python -m autotrust.freeze [--run-id <id>]`"
- Section 4.7: "CLI: `uv run python -m autotrust.export --checkpoint runs/<id>/best.pt --format gguf`"

Neither module has a `__main__.py` or `if __name__ == "__main__"` block. Running `uv run python -m autotrust.freeze` or `uv run python -m autotrust.export` will fail with "No module named autotrust.freeze.__main__".

The README.md documents these CLI commands (lines 78-83), suggesting they should work.

## Evidence
- File: `autotrust/freeze.py` -- no `__main__` block, no `__main__.py` file
- File: `autotrust/export.py` -- no `__main__` block, no `__main__.py` file
- TRD Section 4.2: CLI for freeze
- TRD Section 4.7: CLI for export
- README.md lines 78-83: documents `uv run python -m autotrust.export`

## Suggested Fix
1. Create `autotrust/freeze/__init__.py` + `autotrust/freeze/__main__.py` (or add `if __name__ == "__main__"` block to freeze.py)
2. Create `autotrust/export/__init__.py` + `autotrust/export/__main__.py` (or add to export.py)

For freeze:
```python
if __name__ == "__main__":
    import argparse
    from autotrust.config import load_spec
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    spec = load_spec()
    freeze_teacher(spec, run_id=args.run_id)
```

For export:
```python
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--format", choices=["pytorch", "gguf"], default="pytorch")
    args = parser.parse_args()
    if args.format == "gguf":
        export_gguf(Path(args.checkpoint), Path(args.checkpoint).with_suffix(".gguf"))
    else:
        print(f"Checkpoint at {args.checkpoint} is already in PyTorch format.")
```

## Affected Files
- `autotrust/freeze.py`
- `autotrust/export.py`
