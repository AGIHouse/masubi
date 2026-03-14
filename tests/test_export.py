"""Tests for autotrust.export -- PyTorch/GGUF export pipeline."""

import pytest
import torch
from pathlib import Path

from autotrust.schemas import StudentConfig, CheckpointMeta, MoEConfig


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
def checkpoint_meta(tmp_path):
    return CheckpointMeta(
        stage="dense_baseline",
        experiment_num=5,
        composite=0.82,
        path=tmp_path / "best.pt",
        param_count=500_000,
    )


def test_export_pytorch_creates_file(tmp_path, model, student_config, checkpoint_meta):
    """export_pytorch(model, path) creates a .pt file."""
    from autotrust.export import export_pytorch
    output_path = tmp_path / "model.pt"
    result = export_pytorch(model, student_config, checkpoint_meta, output_path)
    assert result.exists()
    assert result.suffix == ".pt"


def test_export_pytorch_loadable(tmp_path, model, student_config, checkpoint_meta):
    """Saved checkpoint can be loaded and model reproduces same outputs."""
    from autotrust.export import export_pytorch, load_pytorch
    output_path = tmp_path / "model.pt"
    export_pytorch(model, student_config, checkpoint_meta, output_path)

    loaded_model, loaded_config, loaded_meta = load_pytorch(output_path)

    # Verify same outputs
    test_input = torch.randint(0, 1000, (1, 16))
    model.eval()
    loaded_model.eval()
    with torch.no_grad():
        orig_out = model(test_input)
        loaded_out = loaded_model(test_input)
    assert torch.allclose(orig_out["trust_logits"], loaded_out["trust_logits"], atol=1e-5)


def test_export_pytorch_includes_config(tmp_path, model, student_config, checkpoint_meta):
    """Checkpoint includes StudentConfig."""
    from autotrust.export import export_pytorch
    output_path = tmp_path / "model.pt"
    export_pytorch(model, student_config, checkpoint_meta, output_path)

    checkpoint = torch.load(output_path, weights_only=False)
    assert "config" in checkpoint
    restored_config = StudentConfig(**checkpoint["config"])
    assert restored_config.hidden_size == student_config.hidden_size


def test_export_pytorch_includes_meta(tmp_path, model, student_config, checkpoint_meta):
    """Checkpoint includes CheckpointMeta."""
    from autotrust.export import export_pytorch
    output_path = tmp_path / "model.pt"
    export_pytorch(model, student_config, checkpoint_meta, output_path)

    checkpoint = torch.load(output_path, weights_only=False)
    assert "meta" in checkpoint
    restored_meta = CheckpointMeta(**checkpoint["meta"])
    assert restored_meta.composite == 0.82
    assert restored_meta.stage == "dense_baseline"


def test_checkpoint_meta_roundtrip(tmp_path, model, student_config, checkpoint_meta):
    """CheckpointMeta saved in checkpoint deserializes correctly."""
    from autotrust.export import export_pytorch, load_pytorch
    output_path = tmp_path / "model.pt"
    export_pytorch(model, student_config, checkpoint_meta, output_path)

    _, _, loaded_meta = load_pytorch(output_path)
    assert loaded_meta.stage == checkpoint_meta.stage
    assert loaded_meta.experiment_num == checkpoint_meta.experiment_num
    assert loaded_meta.composite == checkpoint_meta.composite
    assert loaded_meta.param_count == checkpoint_meta.param_count


def test_export_gguf_skips_if_unavailable():
    """If llama-cpp-python is not installed, export_gguf() raises ImportError."""
    from autotrust.export import export_gguf
    with pytest.raises(ImportError, match="llama-cpp-python"):
        export_gguf(Path("dummy.pt"), Path("dummy.gguf"))


def test_list_checkpoints(tmp_path, model, student_config):
    """list_checkpoints(run_dir) returns list of CheckpointMeta sorted by composite desc."""
    from autotrust.export import export_pytorch, list_checkpoints

    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()

    # Create two checkpoints with different composites
    for i, composite in enumerate([0.72, 0.85]):
        meta = CheckpointMeta(
            stage="dense_baseline",
            experiment_num=i + 1,
            composite=composite,
            path=checkpoints_dir / f"ckpt_{i}.pt",
            param_count=500_000,
        )
        export_pytorch(model, student_config, meta, checkpoints_dir / f"ckpt_{i}.pt")

    results = list_checkpoints(checkpoints_dir)
    assert len(results) == 2
    assert results[0].composite >= results[1].composite  # sorted desc
    assert results[0].composite == 0.85
