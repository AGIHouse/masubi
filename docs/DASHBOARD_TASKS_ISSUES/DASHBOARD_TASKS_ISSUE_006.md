# Issue 006: Optimization tab missing enhanced composite trend with baseline/best-so-far markers

## Severity
Medium

## Category
Omission

## Description
PRD section 4.2 specifies the Optimization Dashboard should have: "Composite trend line (large) -- same as Live Run but with more context (baseline markers, best-so-far line, improvement rate annotation)."

The implementation reuses the same `charts.composite_trend(metrics)` function from the Live Run tab -- it does not add baseline markers, best-so-far reference line, or improvement rate annotations. The Optimization tab's composite trend is identical to the Live Run tab's, providing no additional analytical value.

## Evidence
- File: `dashboard.py:289` -- `charts.composite_trend(metrics)` -- same as Live Run
- File: `autotrust/dashboard/charts.py:41-67` -- `composite_trend()` has no baseline/best-so-far parameters
- PRD Requirement: Section 4.2 -- "with more context (baseline markers, best-so-far line, improvement rate annotation)"

## Suggested Fix
Create an `enhanced_composite_trend(metrics)` function in `charts.py` that:
1. Adds a horizontal dashed line at the baseline composite (first experiment)
2. Adds a step-wise "best so far" line showing the best composite at each point
3. Annotates the improvement rate (e.g., slope of the best-so-far line)

## Affected Files
- `autotrust/dashboard/charts.py`
- `dashboard.py`
