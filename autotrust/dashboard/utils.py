"""Shared utilities for the dashboard package."""

from __future__ import annotations


def is_kept(result: dict) -> bool:
    """Check if all gates passed (experiment was kept)."""
    gate_results = result.get("gate_results", {})
    return bool(gate_results) and all(gate_results.values())
