# Issue 021: freeze.py and export.py CLI invocation via -m flag does not work

## Severity
Medium

## Category
Bug

## Description
The TRD specifies CLI entry points:
- `uv run python -m autotrust.freeze [--run-id <id>]`
- `uv run python -m autotrust.export --checkpoint <path> --format gguf`

Both `freeze.py` and `export.py` have `if __name__ == "__main__":` blocks, but these only work when the file is invoked directly (e.g., `python autotrust/freeze.py`). The `-m` flag (`python -m autotrust.freeze`) requires a `__main__.py` file inside the `autotrust/freeze/` package or the module to be a package with `__main__.py`.

Since `autotrust/freeze.py` and `autotrust/export.py` are single files (not packages), `python -m autotrust.freeze` will fail with `No module named autotrust.freeze.__main__`.

The existing `autotrust/__main__.py` exists but does not route to freeze or export subcommands.

## Evidence
- File: `autotrust/freeze.py:422-434` -- `if __name__ == "__main__":` block
- File: `autotrust/export.py:160-186` -- `if __name__ == "__main__":` block
- TRD Section 4.2: "CLI: `uv run python -m autotrust.freeze [--run-id <id>]`"
- TRD Section 4.7: "CLI: `uv run python -m autotrust.export --checkpoint ... --format gguf`"

## Suggested Fix
Option A: Add the freeze/export commands to `autotrust/__main__.py`:
```python
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "freeze":
        from autotrust.freeze import main
        main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "export":
        from autotrust.export import main
        main(sys.argv[2:])
```

Option B: Use `python -c "from autotrust.freeze import ..."` or just `python autotrust/freeze.py` in the docs instead.

Option A is preferred to match the TRD CLI spec.

## Affected Files
- `autotrust/__main__.py`
- `autotrust/freeze.py` (extract CLI logic into a `main()` function)
- `autotrust/export.py` (extract CLI logic into a `main()` function)

## Status: Fixed
Extracted CLI logic into `main(argv)` functions in both freeze.py and export.py. Added subcommand routing in `autotrust/__main__.py` supporting `python -m autotrust freeze` and `python -m autotrust export`.
