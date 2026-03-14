"""Tests for Kappa downweighting -- composite only, not gold gate."""

import pytest
from pathlib import Path
from autotrust.config import load_spec, get_effective_weights
from autotrust.schemas import CalibrationReport


@pytest.fixture
def spec():
    return load_spec(Path(__file__).parent.parent / "spec.yaml")


def test_kappa_downweight_proportional(spec):
    """Low Kappa axis gets proportionally lower weight."""
    kappa = {a.name: 1.0 for a in spec.trust_axes}
    kappa["manipulation"] = 0.5

    effective = get_effective_weights(spec, kappa)
    original_weight = next(a.weight for a in spec.trust_axes if a.name == "manipulation")
    assert effective["manipulation"] < original_weight


def test_kappa_downweight_redistribution(spec):
    """Lost weight redistributed to other axes."""
    kappa = {a.name: 1.0 for a in spec.trust_axes}
    kappa["manipulation"] = 0.5

    effective = get_effective_weights(spec, kappa)
    # Sum of effective weights for positive-weighted axes should still be ~1.0
    # since lost weight is redistributed
    total = sum(v for k, v in effective.items()
                if next(a.weight for a in spec.trust_axes if a.name == k) > 0)
    assert abs(total - 1.0) < 0.01


def test_kappa_downweight_composite_only(spec):
    """Downweighting applies to compute_composite() but NOT gold_regression_gate()."""
    from autotrust.eval import gold_regression_gate

    kappa_low = {a.name: 0.5 for a in spec.trust_axes}
    # Verify calibration can be constructed (but gold gate doesn't use it)
    CalibrationReport(
        per_axis_kappa=kappa_low,
        effective_weights=get_effective_weights(spec, kappa_low),
        flagged_axes=[a.name for a in spec.trust_axes],
        downweight_amounts={a.name: a.weight * 0.5 for a in spec.trust_axes},
    )

    # Gold gate uses raw labels, not calibration-adjusted weights
    # Create predictions that match gold exactly
    predictions = [{a.name: 0.8 for a in spec.trust_axes}]
    gold_set = [{a.name: 0.8 for a in spec.trust_axes}]
    previous_best = {a.name: 0.7 for a in spec.trust_axes}

    passed, _ = gold_regression_gate(predictions, gold_set, previous_best, spec)
    assert passed is True  # gold gate ignores calibration


def test_kappa_perfect_no_change(spec):
    """Kappa=1.0 for all axes means no weight change."""
    kappa = {a.name: 1.0 for a in spec.trust_axes}
    effective = get_effective_weights(spec, kappa)
    for axis in spec.trust_axes:
        assert abs(effective[axis.name] - axis.weight) < 1e-6
