"""Tests for starting_train.py -- baseline EmailTrustScorer template."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from datetime import datetime, timezone

from autotrust.config import load_spec
from autotrust.schemas import Email, EmailChain, ScorerOutput, Explanation


@pytest.fixture
def spec():
    return load_spec(Path(__file__).parent.parent / "spec.yaml")


@pytest.fixture
def axis_names(spec):
    return [a.name for a in spec.trust_axes]


@pytest.fixture
def sample_chain(axis_names):
    return EmailChain(
        chain_id="test-001",
        emails=[
            Email(
                from_addr="alice@example.com",
                to_addr="bob@example.com",
                subject="Urgent: Wire Transfer Needed",
                body="Hi Bob, I need you to wire $50,000 to account 12345 immediately. "
                     "This is confidential. Do not tell anyone.",
                timestamp=datetime.now(timezone.utc),
                reply_depth=0,
            ),
        ],
        labels={a: 0.0 for a in axis_names},
        trust_vector={a: 0.0 for a in axis_names},
        composite=0.0,
        flags=[],
    )


@pytest.fixture
def mock_scorer_provider(axis_names):
    """Mock ScoringProvider that returns structured JSON."""
    provider = MagicMock()
    response = json.dumps({
        "trust_vector": {a: 0.5 for a in axis_names},
        "explanation": {
            "reasons": ["phish", "manipulation"],
            "summary": "Suspicious wire transfer request with urgency.",
        },
    })
    provider.score.return_value = response
    return provider


def test_scorer_returns_scorer_output(mock_scorer_provider, sample_chain, spec):
    """score_chain() returns ScorerOutput instance."""
    from starting_train import EmailTrustScorer

    scorer = EmailTrustScorer(provider=mock_scorer_provider, spec=spec)
    output = scorer.score_chain(sample_chain)
    assert isinstance(output, ScorerOutput)


def test_scorer_output_has_trust_vector(mock_scorer_provider, sample_chain, spec, axis_names):
    """Returned ScorerOutput.trust_vector has all spec axis keys."""
    from starting_train import EmailTrustScorer

    scorer = EmailTrustScorer(provider=mock_scorer_provider, spec=spec)
    output = scorer.score_chain(sample_chain)
    assert set(output.trust_vector.keys()) == set(axis_names)


def test_scorer_output_has_explanation(mock_scorer_provider, sample_chain, spec):
    """Returned ScorerOutput.explanation has reasons (list) and summary (str)."""
    from starting_train import EmailTrustScorer

    scorer = EmailTrustScorer(provider=mock_scorer_provider, spec=spec)
    output = scorer.score_chain(sample_chain)
    assert isinstance(output.explanation, Explanation)
    assert isinstance(output.explanation.reasons, list)
    assert isinstance(output.explanation.summary, str)


def test_scorer_batch(mock_scorer_provider, sample_chain, spec):
    """score_batch returns list of ScorerOutput with correct length."""
    from starting_train import EmailTrustScorer

    scorer = EmailTrustScorer(provider=mock_scorer_provider, spec=spec)
    outputs = scorer.score_batch([sample_chain, sample_chain, sample_chain])
    assert len(outputs) == 3
    assert all(isinstance(o, ScorerOutput) for o in outputs)


def test_scorer_reasons_are_strings(mock_scorer_provider, sample_chain, spec):
    """explanation.reasons contains strings."""
    from starting_train import EmailTrustScorer

    scorer = EmailTrustScorer(provider=mock_scorer_provider, spec=spec)
    output = scorer.score_chain(sample_chain)
    assert all(isinstance(r, str) for r in output.explanation.reasons)
