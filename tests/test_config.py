"""Tests for autotrust.config -- typed spec.yaml loader and effective weight computation."""

import pytest
import yaml
from pathlib import Path

from autotrust.config import load_spec, get_spec, get_effective_weights, Spec


@pytest.fixture
def spec_path():
    """Path to the real spec.yaml."""
    return Path(__file__).parent.parent / "spec.yaml"


@pytest.fixture
def spec(spec_path):
    """Load real spec."""
    return load_spec(spec_path)


@pytest.fixture
def make_spec_yaml(tmp_path):
    """Factory to create custom spec.yaml files for testing."""
    def _make(overrides: dict | None = None):
        base = yaml.safe_load(open(Path(__file__).parent.parent / "spec.yaml"))
        if overrides:
            for key, value in overrides.items():
                if isinstance(value, dict) and key in base:
                    base[key].update(value)
                else:
                    base[key] = value
        path = tmp_path / "spec.yaml"
        with open(path, "w") as f:
            yaml.dump(base, f)
        return path
    return _make


def test_load_spec_valid(spec):
    """Loads spec.yaml, returns Spec with correct axis count (10)."""
    assert isinstance(spec, Spec)
    assert len(spec.trust_axes) == 10


def test_axis_weights_sum_to_one(spec):
    """Positive axis weights sum to ~1.0 (within tolerance 0.01)."""
    total = sum(a.weight for a in spec.trust_axes)
    assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected ~1.0"


def test_axis_type_validation(make_spec_yaml):
    """Rejects axes with invalid types (not binary/continuous)."""
    bad_axes = [
        {"name": "phish", "type": "invalid_type", "metric": "f1", "weight": 1.0},
    ]
    path = make_spec_yaml()
    raw = yaml.safe_load(open(path))
    raw["trust_axes"] = bad_axes
    with open(path, "w") as f:
        yaml.dump(raw, f)
    with pytest.raises(Exception):
        load_spec(path)


def test_axis_groups_reference_valid_axes(make_spec_yaml):
    """Rejects spec where axis_groups reference non-existent axes."""
    path = make_spec_yaml()
    raw = yaml.safe_load(open(path))
    raw["axis_groups"]["binary"] = ["nonexistent_axis"]
    with open(path, "w") as f:
        yaml.dump(raw, f)
    with pytest.raises(Exception):
        load_spec(path)


def test_composite_penalties_not_axis_names(make_spec_yaml):
    """Rejects spec where composite_penalty keys match axis names."""
    path = make_spec_yaml()
    raw = yaml.safe_load(open(path))
    raw["composite_penalties"]["phish"] = -0.10  # 'phish' is an axis name
    with open(path, "w") as f:
        yaml.dump(raw, f)
    with pytest.raises(Exception):
        load_spec(path)


def test_get_spec_singleton(spec_path, monkeypatch):
    """Calling get_spec() twice returns same object."""
    import autotrust.config as config_mod
    config_mod._spec = None  # reset
    monkeypatch.setattr(config_mod, "_DEFAULT_SPEC_PATH", spec_path)
    s1 = get_spec()
    s2 = get_spec()
    assert s1 is s2
    config_mod._spec = None  # cleanup


def test_get_effective_weights_no_downweight(spec):
    """With perfect Kappa (1.0 for all axes), effective weights equal original weights."""
    kappa = {a.name: 1.0 for a in spec.trust_axes}
    effective = get_effective_weights(spec, kappa)
    for axis in spec.trust_axes:
        assert abs(effective[axis.name] - axis.weight) < 1e-6, (
            f"{axis.name}: expected {axis.weight}, got {effective[axis.name]}"
        )


def test_get_effective_weights_with_downweight(spec):
    """With low Kappa on one axis, that axis's weight decreases and remainder is redistributed."""
    kappa = {a.name: 1.0 for a in spec.trust_axes}
    kappa["manipulation"] = 0.5  # low Kappa on manipulation (weight=0.13)

    effective = get_effective_weights(spec, kappa)
    # manipulation weight should decrease
    assert effective["manipulation"] < 0.13
    # other positive-weighted axes should increase slightly
    orig_others = sum(a.weight for a in spec.trust_axes if a.name != "manipulation" and a.weight > 0)
    eff_others = sum(effective[a.name] for a in spec.trust_axes if a.name != "manipulation" and a.weight > 0)
    assert eff_others > orig_others


def test_get_effective_weights_zero_weighted_axis(spec):
    """Zero-weighted axes (verify_by_search) stay at 0.0 regardless."""
    kappa = {a.name: 1.0 for a in spec.trust_axes}
    kappa["verify_by_search"] = 0.3  # bad Kappa, but weight is 0.0
    effective = get_effective_weights(spec, kappa)
    assert effective["verify_by_search"] == 0.0
