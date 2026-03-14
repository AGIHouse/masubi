"""Student model for Stage 2 training.

Dense baseline transformer model that scores email text across trust axes,
produces explanation reason tags, and outputs an escalation flag.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from autotrust.schemas import StudentConfig, StudentOutput


class DenseStudent(nn.Module):
    """Dense transformer student model for email trust scoring.

    Takes tokenized email text and produces:
    - trust_logits: per-axis trust scores (num_axes)
    - reason_logits: multi-label reason tag predictions (num_reason_tags)
    - escalate_logit: binary escalation flag (1)
    """

    def __init__(self, config: StudentConfig):
        super().__init__()
        self.config = config

        # Embedding
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_encoding = nn.Embedding(config.max_seq_len, config.hidden_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=max(1, config.hidden_size // 64),
            dim_feedforward=config.hidden_size * 4,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Output heads
        self.trust_head = nn.Linear(config.hidden_size, config.num_axes)
        self.reason_head = nn.Linear(config.hidden_size, config.num_reason_tags)
        self.escalate_head = nn.Linear(config.hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) optional mask (1 = attend, 0 = ignore)

        Returns:
            Dict with trust_logits, reason_logits, escalate_logit tensors.
        """
        batch_size, seq_len = input_ids.shape

        # Clamp positions to max_seq_len
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.clamp(max=self.config.max_seq_len - 1)

        # Embed tokens + positions
        x = self.embedding(input_ids) + self.pos_encoding(positions)

        # Build src_key_padding_mask if attention_mask provided
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0  # True = ignore

        # Transformer encoder
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Pool: mean over sequence dimension
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        # Output heads
        trust_logits = self.trust_head(x)
        reason_logits = self.reason_head(x)
        escalate_logit = self.escalate_head(x)

        return {
            "trust_logits": trust_logits,
            "reason_logits": reason_logits,
            "escalate_logit": escalate_logit,
        }

    @classmethod
    def from_config(cls, config: StudentConfig) -> DenseStudent:
        """Create a DenseStudent model from a StudentConfig."""
        return cls(config)

    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def compute_trust_loss(logits: Tensor, soft_targets: Tensor) -> Tensor:
    """MSE loss between predicted trust scores and teacher soft labels.

    Args:
        logits: (batch, num_axes) raw model output
        soft_targets: (batch, num_axes) teacher soft labels in [0, 1]

    Returns:
        Scalar loss tensor.
    """
    predictions = torch.sigmoid(logits)
    return F.mse_loss(predictions, soft_targets)


def compute_reason_loss(logits: Tensor, tag_targets: Tensor) -> Tensor:
    """Binary cross-entropy for multi-label reason tag prediction.

    Args:
        logits: (batch, num_reason_tags) raw model output
        tag_targets: (batch, num_reason_tags) binary targets

    Returns:
        Scalar loss tensor.
    """
    return F.binary_cross_entropy_with_logits(logits, tag_targets)


def compute_escalate_loss(logit: Tensor, target: Tensor) -> Tensor:
    """Binary cross-entropy for escalation prediction.

    Args:
        logit: (batch, 1) raw model output
        target: (batch, 1) binary target

    Returns:
        Scalar loss tensor.
    """
    return F.binary_cross_entropy_with_logits(logit, target)


def compute_total_loss(
    trust_loss: Tensor,
    reason_loss: Tensor,
    escalate_loss: Tensor,
    trust_weight: float = 1.0,
    reason_weight: float = 0.3,
    escalate_weight: float = 0.2,
) -> Tensor:
    """Weighted combination of all three losses.

    Args:
        trust_loss: scalar trust axis loss
        reason_loss: scalar reason tag loss
        escalate_loss: scalar escalation loss
        trust_weight: weight for trust loss
        reason_weight: weight for reason loss
        escalate_weight: weight for escalation loss

    Returns:
        Scalar combined loss tensor.
    """
    return (
        trust_weight * trust_loss
        + reason_weight * reason_loss
        + escalate_weight * escalate_loss
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict(
    model: DenseStudent,
    input_ids: Tensor,
    axis_names: list[str],
    reason_tag_names: list[str],
    threshold: float = 0.5,
) -> StudentOutput:
    """Run inference and convert logits to StudentOutput schema.

    Args:
        model: trained DenseStudent (or MoEStudent) model
        input_ids: (1, seq_len) tokenized input
        axis_names: list of trust axis names matching model output order
        reason_tag_names: list of reason tag names matching model output order
        threshold: classification threshold for binary outputs

    Returns:
        StudentOutput with trust_vector, reason_tags, escalate.
    """
    model.eval()
    with torch.no_grad():
        output = model(input_ids)

    trust_probs = torch.sigmoid(output["trust_logits"]).squeeze(0)
    reason_probs = torch.sigmoid(output["reason_logits"]).squeeze(0)
    escalate_prob = torch.sigmoid(output["escalate_logit"]).squeeze()

    trust_vector = {
        name: round(trust_probs[i].item(), 4)
        for i, name in enumerate(axis_names)
    }

    reason_tags = [
        reason_tag_names[i]
        for i in range(len(reason_tag_names))
        if reason_probs[i].item() >= threshold
    ]

    return StudentOutput(
        trust_vector=trust_vector,
        reason_tags=reason_tags,
        escalate=escalate_prob.item() >= threshold,
    )
