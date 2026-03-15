# Issue 013: Dead code in export.py -- unused TYPE_CHECKING block

## Severity
Low

## Category
Quality

## Description
In `autotrust/export.py`, `TYPE_CHECKING` is imported on line 10 and used in an empty `if TYPE_CHECKING: pass` block on lines 18-19. This is dead code left over from cleanup that should have been caught in TASK_012.

## Evidence
- File: `autotrust/export.py:10` -- `from typing import TYPE_CHECKING`
- File: `autotrust/export.py:18-19` -- `if TYPE_CHECKING: pass`

## Suggested Fix
Remove the `TYPE_CHECKING` import and the empty guard block:
```python
# Delete these lines:
from typing import TYPE_CHECKING
...
if TYPE_CHECKING:
    pass
```

## Affected Files
- `autotrust/export.py`
