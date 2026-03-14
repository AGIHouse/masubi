"""Tests for explanation quality and explanation gate logic."""

import pytest
from pathlib import Path
from autotrust.config import load_spec
from autotrust.schemas import Explanation


@pytest.fixture
def spec():
    return load_spec(Path(__file__).parent.parent / "spec.yaml")


def test_explanation_quality_all_flagged_referenced(spec):
    """All flagged axes in reasons -> quality=1.0."""
    from autotrust.eval import explanation_quality

    # predictions with phish=0.8, manipulation=0.7 (both above flag_threshold 0.5)
    predictions = [{"phish": 0.8, "manipulation": 0.7, "truthfulness": 0.2}]
    explanations = [Explanation(
        reasons=["phish", "manipulation"],
        summary="Phishing with manipulation.",
    )]

    quality = explanation_quality(explanations, predictions, spec)
    assert quality == 1.0


def test_explanation_quality_partial_reference(spec):
    """1 of 3 flagged axes referenced -> quality=0.33."""
    from autotrust.eval import explanation_quality

    predictions = [{"phish": 0.8, "manipulation": 0.7, "deceit": 0.9}]
    # Fill in all other axes below threshold
    for a in spec.trust_axes:
        if a.name not in predictions[0]:
            predictions[0][a.name] = 0.1

    explanations = [Explanation(
        reasons=["phish"],  # only 1 of 3 flagged
        summary="Phishing detected.",
    )]

    quality = explanation_quality(explanations, predictions, spec)
    assert abs(quality - 1.0 / 3.0) < 0.05


def test_explanation_quality_no_flags(spec):
    """No axes above threshold -> quality=1.0."""
    from autotrust.eval import explanation_quality

    predictions = [{a.name: 0.1 for a in spec.trust_axes}]
    explanations = [Explanation(reasons=[], summary="Clean email.")]

    quality = explanation_quality(explanations, predictions, spec)
    assert quality == 1.0


def test_explanation_gate_warn_mode(spec):
    """Before baseline, logs but always passes."""
    from autotrust.eval import explanation_gate

    passed, mode = explanation_gate(0.1, spec, has_baseline=False)
    assert passed is True
    assert mode == "warn"


def test_explanation_gate_gate_mode_passes(spec):
    """After baseline, quality >= threshold -> passes."""
    from autotrust.eval import explanation_gate

    passed, mode = explanation_gate(0.8, spec, has_baseline=True)
    assert passed is True
    assert mode == "gate"


def test_explanation_gate_gate_mode_blocks(spec):
    """After baseline, quality < threshold -> blocks."""
    from autotrust.eval import explanation_gate

    passed, mode = explanation_gate(0.1, spec, has_baseline=True)
    assert passed is False
    assert mode == "gate"
