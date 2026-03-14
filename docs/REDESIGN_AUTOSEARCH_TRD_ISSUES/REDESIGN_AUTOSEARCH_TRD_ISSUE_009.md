# Issue 009: Dashboard stage indicator and checkpoint UI not implemented

## Severity
Medium

## Category
Omission

## Description
The TRD section 4.9 requires:
1. "Stage indicator in Live Run tab (Stage 1: Prompt Optimization / Stage 2: Model Training)"
2. "Checkpoint list with export buttons"
3. "Model size / latency stats for production readiness"

The TASK_010 Review Notes confirm: "Dashboard UI integration (stage indicator, checkpoint list) deferred to integration phase."

Only the chart functions (`training_loss`, `param_count_timeline`, `expert_utilization`) were implemented. The Gradio dashboard (`dashboard.py`) was not updated to include these charts or the required UI elements.

## Evidence
- File: `autotrust/dashboard/charts.py` -- chart functions exist but not integrated into dashboard
- File: `dashboard.py` -- not modified to include stage indicator, checkpoint list, or Stage 2 charts
- TRD Section 4.9: "Stage indicator", "Checkpoint list with export buttons", "Model size / latency stats"
- TASK_010 Review Notes: "Dashboard UI integration deferred"

## Suggested Fix
1. Add stage indicator `gr.Textbox` to Live Run tab in `dashboard.py`
2. Add conditional rendering of Stage 2 charts when training metrics exist
3. Add checkpoint management section using `list_checkpoints()` from `export.py`
4. Add export buttons that call `export_pytorch()` / `export_gguf()`
5. Update `poll_update()` to refresh stage indicator and training charts

## Affected Files
- `dashboard.py`
- `tests/test_dashboard_integration.py`
