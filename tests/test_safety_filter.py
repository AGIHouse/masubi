"""Tests for data.py safety filter and calibration logic."""

import pytest
from pathlib import Path
from autotrust.config import load_spec
from autotrust.schemas import Email
from datetime import datetime, timezone


@pytest.fixture
def spec():
    return load_spec(Path(__file__).parent.parent / "spec.yaml")


@pytest.fixture
def clean_email():
    return Email(
        from_addr="alice@example.com",
        to_addr="bob@example.com",
        subject="Meeting tomorrow",
        body="Hi Bob, let's meet at 3pm to discuss the project.",
        timestamp=datetime.now(timezone.utc),
        reply_depth=0,
    )


@pytest.fixture
def operational_email():
    return Email(
        from_addr="attacker@evil.com",
        to_addr="victim@example.com",
        subject="Instructions",
        body="Here are the step-by-step instructions to exploit the vulnerability: "
             "1. Download the payload from malware.exe "
             "2. Execute as administrator "
             "3. Use the reverse shell to connect to C2 server",
        timestamp=datetime.now(timezone.utc),
        reply_depth=0,
    )


@pytest.fixture
def real_brand_email():
    return Email(
        from_addr="noreply@paypal.com",
        to_addr="user@gmail.com",
        subject="Account verification",
        body="Dear PayPal user, please verify your account at Microsoft portal.",
        timestamp=datetime.now(timezone.utc),
        reply_depth=0,
    )


@pytest.fixture
def structural_phish_email():
    return Email(
        from_addr="support@example-corp.com",
        to_addr="user@example.com",
        subject="Urgent: Account suspended",
        body="Your account has been suspended. Click here to verify your identity immediately or lose access.",
        timestamp=datetime.now(timezone.utc),
        reply_depth=0,
    )


def test_safety_filter_blocks_operational_instructions(operational_email, spec):
    """Emails with operational attack instructions are rejected."""
    from autotrust.data import safety_filter
    assert safety_filter(operational_email, is_synth=True, spec=spec) is False


def test_safety_filter_allows_structural_malicious(structural_phish_email, spec):
    """Structurally malicious emails (phishing patterns without real ops) pass."""
    from autotrust.data import safety_filter
    assert safety_filter(structural_phish_email, is_synth=True, spec=spec) is True


def test_safety_filter_placeholder_only(real_brand_email, spec):
    """Synth emails use placeholder brand names, not real brands."""
    from autotrust.data import safety_filter
    # Real brand in synth data -> rejected
    assert safety_filter(real_brand_email, is_synth=True, spec=spec) is False


def test_safety_filter_real_brands_in_eval(real_brand_email, spec):
    """Eval set allows real brand names."""
    from autotrust.data import safety_filter
    assert safety_filter(real_brand_email, is_synth=False, spec=spec) is True


def test_calibrate_judge_computes_kappa():
    """With known annotator scores, verify Cohen's Kappa computation is correct."""
    from autotrust.data import compute_cohen_kappa

    # Perfect agreement
    annotator_1 = [1.0, 0.0, 1.0, 0.0, 1.0]
    annotator_2 = [1.0, 0.0, 1.0, 0.0, 1.0]
    kappa = compute_cohen_kappa(annotator_1, annotator_2)
    assert kappa == 1.0


def test_calibrate_judge_flags_low_kappa(spec):
    """Axes below min_gold_kappa are flagged."""
    from autotrust.data import flag_low_kappa_axes

    kappa_per_axis = {a.name: 0.9 for a in spec.trust_axes}
    kappa_per_axis["deceit"] = 0.5  # below min_gold_kappa of 0.7

    flagged = flag_low_kappa_axes(kappa_per_axis, spec)
    assert "deceit" in flagged
    assert "phish" not in flagged
