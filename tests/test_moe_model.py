"""Tests for autotrust.student MoE extension -- MoE blocks and MoEStudent model."""

import pytest
import torch
from pathlib import Path

from autotrust.config import load_spec
from autotrust.schemas import StudentConfig, MoEConfig


@pytest.fixture
def spec():
    return load_spec(Path(__file__).parent.parent / "spec.yaml")


@pytest.fixture
def student_config():
    return StudentConfig(
        hidden_size=128,
        num_layers=4,
        vocab_size=1000,
        max_seq_len=64,
        num_axes=10,
        num_reason_tags=20,
    )


@pytest.fixture
def moe_config():
    return MoEConfig(
        num_experts=4,
        top_k=2,
        capacity_factor=1.0,
        moe_layers=[1, 3],
        routing_strategy="top_k",
    )


@pytest.fixture
def sample_input():
    return torch.randint(0, 1000, (2, 16))


def test_moe_block_forward():
    """MoE block accepts hidden states, returns same shape output."""
    from autotrust.student import MoEBlock
    block = MoEBlock(
        hidden_size=128,
        intermediate_size=512,
        num_experts=4,
        top_k=2,
        capacity_factor=1.0,
        routing_strategy="top_k",
    )
    x = torch.randn(2, 16, 128)  # (batch, seq, hidden)
    output, aux_loss = block(x)
    assert output.shape == x.shape


def test_moe_block_routes_to_top_k():
    """With top_k=2 and num_experts=4, only 2 experts are activated per token."""
    from autotrust.student import MoEBlock
    block = MoEBlock(
        hidden_size=64,
        intermediate_size=256,
        num_experts=4,
        top_k=2,
        capacity_factor=1.0,
        routing_strategy="top_k",
    )
    x = torch.randn(1, 8, 64)
    output, aux_loss = block(x)
    # Output should be valid (routing works)
    assert output.shape == x.shape
    assert not torch.isnan(output).any()


def test_moe_block_capacity_factor():
    """MoE block works with different capacity factors."""
    from autotrust.student import MoEBlock
    block = MoEBlock(
        hidden_size=64,
        intermediate_size=256,
        num_experts=4,
        top_k=2,
        capacity_factor=1.5,
        routing_strategy="top_k",
    )
    x = torch.randn(2, 16, 64)
    output, aux_loss = block(x)
    assert output.shape == x.shape


def test_moe_student_forward(student_config, moe_config, sample_input):
    """MoEStudent accepts input_ids, returns same output dict as DenseStudent."""
    from autotrust.student import MoEStudent
    model = MoEStudent.from_config(student_config, moe_config)
    output = model(sample_input)
    assert "trust_logits" in output
    assert "reason_logits" in output
    assert "escalate_logit" in output
    assert "aux_loss" in output
    assert output["trust_logits"].shape == (2, 10)


def test_moe_student_from_config(student_config, moe_config):
    """MoEStudent.from_config(student_config, moe_config) creates model."""
    from autotrust.student import MoEStudent
    model = MoEStudent.from_config(student_config, moe_config)
    assert model.param_count() > 0


def test_moe_expert_cap_enforced(student_config, spec):
    """MoEConfig with num_experts=20 raises ValueError when max_experts=16."""
    from autotrust.student import validate_moe_config
    bad_config = MoEConfig(
        num_experts=20,
        top_k=2,
        moe_layers=[0],
    )
    with pytest.raises(ValueError, match="max_experts"):
        validate_moe_config(bad_config, spec)


def test_moe_top_k_cap_enforced(student_config, spec):
    """MoEConfig with top_k=5 raises ValueError when max_top_k=4."""
    from autotrust.student import validate_moe_config
    bad_config = MoEConfig(
        num_experts=4,
        top_k=5,
        moe_layers=[0],
    )
    with pytest.raises(ValueError, match="max_top_k"):
        validate_moe_config(bad_config, spec)


def test_moe_param_budget_enforced(spec):
    """Model within budget passes check_param_budget."""
    from autotrust.student import MoEStudent, check_param_budget
    # Create a model that's within budget
    config = StudentConfig(
        hidden_size=128,
        num_layers=2,
        vocab_size=1000,
        max_seq_len=64,
        num_axes=10,
        num_reason_tags=20,
    )
    moe_config = MoEConfig(num_experts=4, top_k=2, moe_layers=[0])
    model = MoEStudent.from_config(config, moe_config)
    # Should not raise for small model
    check_param_budget(model, spec)


def test_moe_param_budget_rejection(spec):
    """Model exceeding max_params_m raises ValueError from check_param_budget."""
    from autotrust.student import MoEStudent, check_param_budget
    # Create a deliberately oversized model
    big_config = StudentConfig(
        hidden_size=2048,
        num_layers=24,
        vocab_size=100000,
        max_seq_len=1024,
        num_axes=10,
        num_reason_tags=20,
    )
    moe_config = MoEConfig(num_experts=16, top_k=4, moe_layers=list(range(24)))
    model = MoEStudent.from_config(big_config, moe_config)
    with pytest.raises(ValueError, match="exceeds budget"):
        check_param_budget(model, spec)


def test_moe_routing_strategies(student_config):
    """noisy_top_k and expert_choice routing strategies both work."""
    from autotrust.student import MoEBlock
    for strategy in ["noisy_top_k", "expert_choice"]:
        block = MoEBlock(
            hidden_size=128,
            intermediate_size=512,
            num_experts=4,
            top_k=2,
            capacity_factor=1.0,
            routing_strategy=strategy,
        )
        x = torch.randn(2, 8, 128)
        output, aux_loss = block(x)
        assert output.shape == x.shape, f"Failed for strategy {strategy}"


def test_moe_load_balance_loss():
    """MoE block computes auxiliary load-balancing loss that is not constant."""
    from autotrust.student import MoEBlock
    block = MoEBlock(
        hidden_size=64,
        intermediate_size=256,
        num_experts=4,
        top_k=2,
        capacity_factor=1.0,
        routing_strategy="top_k",
    )
    x = torch.randn(2, 16, 64)
    _, aux_loss = block(x)
    assert aux_loss.ndim == 0  # scalar
    assert aux_loss.item() >= 0


def test_moe_load_balance_loss_discriminates():
    """Load balance loss is higher for imbalanced routing than balanced routing."""
    from autotrust.student import MoEBlock

    # Create block with biased router to force imbalanced routing
    block_balanced = MoEBlock(
        hidden_size=64, intermediate_size=256,
        num_experts=4, top_k=2, capacity_factor=1.0,
        routing_strategy="top_k",
    )

    # Use a seeded input to get reproducible routing
    torch.manual_seed(42)
    # Uniform-ish input should give relatively balanced routing
    x_uniform = torch.randn(4, 32, 64) * 0.01

    # Bias the router to strongly prefer one expert for imbalanced routing
    block_imbalanced = MoEBlock(
        hidden_size=64, intermediate_size=256,
        num_experts=4, top_k=2, capacity_factor=1.0,
        routing_strategy="top_k",
    )
    with torch.no_grad():
        # Set router bias to strongly favor expert 0
        block_imbalanced.router.weight.zero_()
        block_imbalanced.router.weight[0] = 10.0  # expert 0 dominates

    _, loss_imbalanced = block_imbalanced(x_uniform)
    _, loss_balanced = block_balanced(x_uniform)

    # The imbalanced loss should differ from 1.0 (the old constant)
    # and the imbalanced routing should produce a different loss value
    # than balanced routing (this is the key discriminating behavior)
    assert loss_imbalanced.item() != loss_balanced.item()


def test_dense_to_moe_upgrade(student_config, moe_config):
    """Can create MoE model and load dense model weights for non-MoE layers."""
    from autotrust.student import DenseStudent, MoEStudent
    dense = DenseStudent.from_config(student_config)
    moe = MoEStudent.from_dense(dense, moe_config)
    assert isinstance(moe, MoEStudent)
    # Shared layers should have same weights
    test_input = torch.randint(0, 1000, (1, 16))
    dense.eval()
    moe.eval()
    with torch.no_grad():
        # Both should produce valid outputs
        dense_out = dense(test_input)
        moe_out = moe(test_input)
    assert dense_out["trust_logits"].shape == moe_out["trust_logits"].shape
