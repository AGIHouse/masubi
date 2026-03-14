"""Tests for gold regression gate -- raw labels, no downweighting, all axes."""

import pytest
from pathlib import Path
from autotrust.config import load_spec


@pytest.fixture
def spec():
    return load_spec(Path(__file__).parent.parent / "spec.yaml")


@pytest.fixture
def axis_names(spec):
    return [a.name for a in spec.trust_axes]


def test_gold_gate_passes_when_no_regression(spec, axis_names):
    """No axis degrades -> passes."""
    from autotrust.eval import gold_regression_gate

    predictions = [{a: 0.8 for a in axis_names}]
    gold_set = [{a: 0.8 for a in axis_names}]
    previous_best = {a: 0.7 for a in axis_names}

    passed, deltas = gold_regression_gate(predictions, gold_set, previous_best, spec)
    assert passed is True


def test_gold_gate_rejects_any_axis_regression(spec, axis_names):
    """Single axis degrades -> veto."""
    from autotrust.eval import gold_regression_gate

    # Predictions disagree with gold on truthfulness, creating regression
    predictions = [{a: 0.8 for a in axis_names}]
    gold_set = [{a: 0.8 for a in axis_names}]
    # Make truthfulness (continuous, agreement) mismatch to get low score
    predictions[0]["truthfulness"] = 0.2  # diverges from gold 0.8 -> agreement ~0.4
    previous_best = {a: 0.5 for a in axis_names}
    previous_best["truthfulness"] = 0.95  # previous was much better -> regression

    passed, deltas = gold_regression_gate(predictions, gold_set, previous_best, spec)
    assert passed is False


def test_gold_gate_uses_raw_labels(spec, axis_names):
    """Gold gate ignores Kappa downweighting."""
    from autotrust.eval import gold_regression_gate

    # Same test as above -- gold gate doesn't take calibration as an argument
    predictions = [{a: 0.9 for a in axis_names}]
    gold_set = [{a: 0.9 for a in axis_names}]
    previous_best = {a: 0.85 for a in axis_names}

    passed, _ = gold_regression_gate(predictions, gold_set, previous_best, spec)
    assert passed is True


def test_gold_gate_zero_weighted_axis_still_vetoes(spec, axis_names):
    """verify_by_search (weight=0.0) regression still triggers veto."""
    from autotrust.eval import gold_regression_gate

    # Make verify_by_search predictions wrong vs gold to create regression
    predictions = [{a: 0.8 for a in axis_names}]
    gold_set = [{a: 0.8 for a in axis_names}]
    # verify_by_search is binary: pred=0.8(->1), gold=0.8(->1) would match
    # Make prediction wrong: pred=0.1(->0) vs gold=0.8(->1)
    predictions[0]["verify_by_search"] = 0.1
    previous_best = {a: 0.5 for a in axis_names}
    previous_best["verify_by_search"] = 1.0  # previous was perfect -> regression

    passed, deltas = gold_regression_gate(predictions, gold_set, previous_best, spec)
    assert passed is False


def test_gold_gate_overrides_composite_improvement(spec, axis_names):
    """Composite improves but gold gate vetoes -> discard."""
    from autotrust.eval import keep_or_discard

    # composite_improved=True but gold_ok=False
    assert keep_or_discard(True, False, True) is False
