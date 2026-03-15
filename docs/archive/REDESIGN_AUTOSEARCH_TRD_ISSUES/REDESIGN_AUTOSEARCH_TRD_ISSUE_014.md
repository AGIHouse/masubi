# Issue 014: spec.yaml explanation config not reconciled with PRD

## Severity
Low

## Category
Omission

## Description
The TRD section 4.1 notes: "The current `explanation` section uses `mode` and `gate_after_baseline` but the PRD uses `gate_enabled: true`. These need reconciliation."

This reconciliation was not performed. The `explanation` section still uses the original field names (`mode`, `gate_after_baseline`) rather than the PRD's `gate_enabled`. While the functionality works correctly with the current fields, the TRD explicitly called out this as something that "needs reconciliation."

This is a minor scope gap since the existing explanation gate works correctly, and the PRD's `gate_enabled` is less expressive than the current `mode` + `gate_after_baseline` combination.

## Evidence
- File: `spec.yaml:107-111` -- `explanation` section with `mode` and `gate_after_baseline`
- File: `autotrust/config.py:77-82` -- `ExplanationConfig` with `mode`, `gate_after_baseline`
- TRD Section 4.1: "These need reconciliation"

## Suggested Fix
Either:
1. Add `gate_enabled` to `ExplanationConfig` as a computed property: `gate_enabled = mode == "warn_then_gate" and gate_after_baseline`
2. Or document in the TRD that this reconciliation was intentionally deferred and the existing fields are preferred

## Affected Files
- `spec.yaml`
- `autotrust/config.py`
