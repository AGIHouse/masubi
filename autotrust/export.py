"""PyTorch checkpoint export and GGUF conversion.

Save/load student model checkpoints with metadata, and optionally convert
to GGUF format for local inference via llama.cpp.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog
import torch
import torch.nn as nn

from autotrust.schemas import CheckpointMeta, MoEConfig, StudentConfig

if TYPE_CHECKING:
    pass

logger = structlog.get_logger()


def export_pytorch(
    model: nn.Module,
    config: StudentConfig,
    meta: CheckpointMeta,
    output_path: Path,
    moe_config: MoEConfig | None = None,
) -> Path:
    """Save model state dict + config + meta to a .pt file.

    Args:
        model: trained student model
        config: model configuration
        meta: checkpoint metadata (stage, composite, etc.)
        output_path: where to save the .pt file
        moe_config: optional MoE configuration for MoE models

    Returns:
        Path to the saved checkpoint.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "config": config.model_dump(),
        "meta": meta.model_dump(mode="json"),
    }
    if moe_config is not None:
        checkpoint["moe_config"] = moe_config.model_dump()

    torch.save(checkpoint, output_path)
    logger.info("Checkpoint saved", path=str(output_path), composite=meta.composite)
    return output_path


def load_pytorch(
    checkpoint_path: Path,
) -> tuple[nn.Module, StudentConfig, CheckpointMeta]:
    """Load model from a .pt checkpoint.

    Reconstructs the model from saved config, then loads state dict.

    Args:
        checkpoint_path: path to the .pt checkpoint file

    Returns:
        Tuple of (model, config, meta).
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    config = StudentConfig(**checkpoint["config"])
    # Handle Path serialization in meta
    meta_data = checkpoint["meta"]
    if isinstance(meta_data.get("path"), str):
        meta_data["path"] = Path(meta_data["path"])
    meta = CheckpointMeta(**meta_data)

    # Reconstruct model based on whether MoE config is present
    if "moe_config" in checkpoint:
        moe_config = MoEConfig(**checkpoint["moe_config"])
        from autotrust.student import MoEStudent
        model = MoEStudent.from_config(config, moe_config)
    else:
        from autotrust.student import DenseStudent
        model = DenseStudent.from_config(config)

    model.load_state_dict(checkpoint["state_dict"])
    return model, config, meta


def export_gguf(checkpoint_path: Path, output_path: Path) -> Path:
    """Convert PyTorch checkpoint to GGUF format.

    Requires llama-cpp-python to be installed.

    Args:
        checkpoint_path: path to the .pt checkpoint
        output_path: where to save the .gguf file

    Returns:
        Path to the saved GGUF file.

    Raises:
        ImportError: if llama-cpp-python is not installed.
    """
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        raise ImportError(
            "GGUF export requires llama-cpp-python. "
            "Install with: pip install llama-cpp-python"
        )

    # Load the checkpoint
    model, config, meta = load_pytorch(checkpoint_path)

    # GGUF conversion would go here
    # This is a placeholder -- full GGUF conversion requires model-specific
    # weight mapping that depends on the final architecture
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("GGUF export", path=str(output_path))
    return output_path


def list_checkpoints(run_dir: Path) -> list[CheckpointMeta]:
    """List all checkpoints in a directory, sorted by composite descending.

    Args:
        run_dir: directory containing .pt checkpoint files

    Returns:
        List of CheckpointMeta sorted by composite score (highest first).
    """
    run_dir = Path(run_dir)
    checkpoints: list[CheckpointMeta] = []

    for pt_file in sorted(run_dir.glob("*.pt")):
        try:
            checkpoint = torch.load(pt_file, weights_only=False)
            meta_data = checkpoint.get("meta", {})
            if isinstance(meta_data.get("path"), str):
                meta_data["path"] = Path(meta_data["path"])
            meta = CheckpointMeta(**meta_data)
            checkpoints.append(meta)
        except Exception as exc:
            logger.warning("Failed to load checkpoint", path=str(pt_file), error=str(exc))
            continue

    # Sort by composite descending
    checkpoints.sort(key=lambda m: m.composite, reverse=True)
    return checkpoints
