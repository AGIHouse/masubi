"""Tests for autotrust.student -- Dense baseline student model."""

import pytest
import torch
from pathlib import Path

from autotrust.config import load_spec
from autotrust.schemas import StudentConfig, StudentOutput


@pytest.fixture
def spec():
    return load_spec(Path(__file__).parent.parent / "spec.yaml")


@pytest.fixture
def student_config():
    return StudentConfig(
        hidden_size=128,
        num_layers=2,
        vocab_size=1000,
        max_seq_len=64,
        num_axes=10,
        num_reason_tags=20,
    )


@pytest.fixture
def model(student_config):
    from autotrust.student import DenseStudent
    return DenseStudent.from_config(student_config)


@pytest.fixture
def sample_input():
    """Batch of 2 sequences, each length 16."""
    return torch.randint(0, 1000, (2, 16))


def test_dense_model_forward_pass(model, sample_input):
    """Model accepts input_ids tensor, returns dict with trust_logits, reason_logits, escalate_logit."""
    output = model(sample_input)
    assert "trust_logits" in output
    assert "reason_logits" in output
    assert "escalate_logit" in output


def test_dense_model_output_shapes(model, sample_input):
    """trust_logits shape is (batch, 10), reason_logits is (batch, 20), escalate_logit is (batch, 1)."""
    output = model(sample_input)
    assert output["trust_logits"].shape == (2, 10)
    assert output["reason_logits"].shape == (2, 20)
    assert output["escalate_logit"].shape == (2, 1)


def test_dense_model_param_count_within_budget(model, spec):
    """Total params <= spec.stage2.max_params_m * 1e6."""
    max_params = spec.stage2.max_params_m * 1_000_000
    assert model.param_count() <= max_params


def test_dense_model_from_config(student_config):
    """DenseStudent.from_config(StudentConfig) creates model correctly."""
    from autotrust.student import DenseStudent
    model = DenseStudent.from_config(student_config)
    assert isinstance(model, DenseStudent)
    assert model.param_count() > 0


def test_trust_loss_soft_labels():
    """compute_trust_loss(logits, soft_targets) returns scalar loss."""
    from autotrust.student import compute_trust_loss
    logits = torch.randn(4, 10)
    targets = torch.rand(4, 10)
    loss = compute_trust_loss(logits, targets)
    assert loss.ndim == 0  # scalar
    assert loss.item() >= 0


def test_reason_tag_loss():
    """compute_reason_loss(logits, tag_targets) returns scalar BCE loss."""
    from autotrust.student import compute_reason_loss
    logits = torch.randn(4, 20)
    targets = torch.zeros(4, 20)
    targets[:, :5] = 1.0
    loss = compute_reason_loss(logits, targets)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_escalate_loss():
    """compute_escalate_loss(logit, target) returns scalar BCE loss."""
    from autotrust.student import compute_escalate_loss
    logit = torch.randn(4, 1)
    target = torch.zeros(4, 1)
    loss = compute_escalate_loss(logit, target)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_combined_loss_weighted():
    """compute_total_loss() combines all three with configurable weights."""
    from autotrust.student import compute_total_loss
    trust_loss = torch.tensor(1.0)
    reason_loss = torch.tensor(0.5)
    escalate_loss = torch.tensor(0.3)
    total = compute_total_loss(trust_loss, reason_loss, escalate_loss,
                                trust_weight=1.0, reason_weight=0.3, escalate_weight=0.2)
    expected = 1.0 * 1.0 + 0.3 * 0.5 + 0.2 * 0.3
    assert abs(total.item() - expected) < 1e-5
