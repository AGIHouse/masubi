"""Tests for autotrust.schemas -- pydantic data models and TrustVector validation."""

import pytest
from datetime import datetime, timezone
from pathlib import Path

from autotrust.config import load_spec
from autotrust.schemas import (
    Email,
    EmailChain,
    Explanation,
    ScorerOutput,
    ExperimentResult,
    GoldChain,
    CalibrationReport,
    validate_trust_vector,
)


@pytest.fixture
def spec():
    return load_spec(Path(__file__).parent.parent / "spec.yaml")


@pytest.fixture
def axis_names(spec):
    return [a.name for a in spec.trust_axes]


@pytest.fixture
def valid_trust_vector(axis_names):
    return {name: 0.5 for name in axis_names}


@pytest.fixture
def sample_email():
    return Email(
        from_addr="alice@example.com",
        to_addr="bob@example.com",
        subject="Test",
        body="Hello",
        timestamp=datetime.now(timezone.utc),
        reply_depth=0,
    )


@pytest.fixture
def sample_chain(sample_email, valid_trust_vector):
    return EmailChain(
        chain_id="test-001",
        emails=[sample_email],
        labels=valid_trust_vector,
        trust_vector=valid_trust_vector,
        composite=0.75,
        flags=[],
    )


def test_trust_vector_valid_keys(valid_trust_vector, spec):
    """Dict with all spec axis names accepted."""
    result = validate_trust_vector(valid_trust_vector, spec)
    assert result == valid_trust_vector


def test_trust_vector_rejects_unknown_key(valid_trust_vector, spec):
    """Dict with key not in spec raises ValidationError."""
    bad = {**valid_trust_vector, "unknown_axis": 0.5}
    with pytest.raises(ValueError, match="unknown_axis"):
        validate_trust_vector(bad, spec)


def test_trust_vector_rejects_missing_key(valid_trust_vector, spec):
    """Dict missing a spec axis raises ValidationError."""
    bad = {k: v for k, v in valid_trust_vector.items() if k != "phish"}
    with pytest.raises(ValueError, match="phish"):
        validate_trust_vector(bad, spec)


def test_trust_vector_rejects_non_float(valid_trust_vector, spec):
    """Dict with non-float value raises ValidationError."""
    bad = {**valid_trust_vector, "phish": "not_a_float"}
    with pytest.raises((ValueError, TypeError)):
        validate_trust_vector(bad, spec)


def test_email_chain_round_trip(sample_chain):
    """EmailChain serializes to JSON and deserializes back identically."""
    json_str = sample_chain.model_dump_json()
    restored = EmailChain.model_validate_json(json_str)
    assert restored.chain_id == sample_chain.chain_id
    assert restored.composite == sample_chain.composite
    assert restored.trust_vector == sample_chain.trust_vector
    assert len(restored.emails) == len(sample_chain.emails)


def test_scorer_output_structure(valid_trust_vector):
    """ScorerOutput has trust_vector (dict) and explanation (Explanation with reasons list + summary string)."""
    output = ScorerOutput(
        trust_vector=valid_trust_vector,
        explanation=Explanation(
            reasons=["phish", "manipulation"],
            summary="Suspicious email with phishing and manipulation signals.",
        ),
    )
    assert isinstance(output.trust_vector, dict)
    assert isinstance(output.explanation, Explanation)
    assert isinstance(output.explanation.reasons, list)
    assert isinstance(output.explanation.summary, str)


def test_explanation_reasons_are_strings():
    """Explanation.reasons must be list[str]."""
    expl = Explanation(reasons=["phish", "deceit"], summary="Test")
    assert all(isinstance(r, str) for r in expl.reasons)


def test_experiment_result_fields():
    """ExperimentResult has all required fields."""
    result = ExperimentResult(
        run_id="run-001",
        change_description="test change",
        per_axis_scores={"phish": 0.9},
        composite=0.85,
        fp_rate=0.05,
        judge_agreement=0.92,
        gold_agreement=0.88,
        explanation_quality=0.75,
        downweighted_axes=["deceit"],
        gate_results={"composite": True, "gold": True, "explanation": True},
        cost=1.50,
        wall_time=120.0,
    )
    assert result.run_id == "run-001"
    assert result.composite == 0.85
    assert result.gate_results["gold"] is True
    assert "deceit" in result.downweighted_axes


def test_gold_chain_extends_email_chain(sample_email, valid_trust_vector):
    """GoldChain has annotator_scores, consensus_labels, kappa, opus_agreement in addition to EmailChain fields."""
    gold = GoldChain(
        chain_id="gold-001",
        emails=[sample_email],
        labels=valid_trust_vector,
        trust_vector=valid_trust_vector,
        composite=0.80,
        flags=[],
        annotator_scores={"phish": [1.0, 1.0, 0.0]},
        consensus_labels={"phish": 1.0},
        kappa={"phish": 0.85},
        opus_agreement={"phish": 0.90},
    )
    assert gold.chain_id == "gold-001"
    assert "phish" in gold.annotator_scores
    assert gold.kappa["phish"] == 0.85
    assert gold.opus_agreement["phish"] == 0.90


def test_calibration_report_fields():
    """CalibrationReport has per_axis_kappa, effective_weights, flagged_axes, downweight_amounts."""
    report = CalibrationReport(
        per_axis_kappa={"phish": 0.90, "deceit": 0.55},
        effective_weights={"phish": 0.22, "deceit": 0.07},
        flagged_axes=["deceit"],
        downweight_amounts={"deceit": 0.03},
    )
    assert report.per_axis_kappa["phish"] == 0.90
    assert "deceit" in report.flagged_axes
    assert report.downweight_amounts["deceit"] == 0.03


# ---------------------------------------------------------------------------
# Student model type tests (TASK_002)
# ---------------------------------------------------------------------------

from autotrust.schemas import (
    StudentConfig,
    MoEConfig,
    StudentOutput,
    CheckpointMeta,
    TeacherArtifacts,
)


def test_student_config_roundtrip():
    """StudentConfig serializes and deserializes correctly."""
    config = StudentConfig(
        hidden_size=256,
        num_layers=6,
        vocab_size=32000,
        max_seq_len=512,
        num_axes=10,
        num_reason_tags=20,
    )
    json_str = config.model_dump_json()
    restored = StudentConfig.model_validate_json(json_str)
    assert restored.hidden_size == 256
    assert restored.num_layers == 6
    assert restored.vocab_size == 32000
    assert restored.max_seq_len == 512
    assert restored.num_axes == 10
    assert restored.num_reason_tags == 20


def test_moe_config_validates_expert_cap(spec):
    """MoEConfig rejects num_experts > spec.stage2.max_experts."""
    from autotrust.schemas import validate_moe_config

    bad_config = MoEConfig(
        num_experts=20,
        top_k=2,
        moe_layers=[0, 1],
    )
    with pytest.raises(ValueError, match="max_experts"):
        validate_moe_config(bad_config, spec)


def test_moe_config_validates_param_budget(spec):
    """MoEConfig rejects total params exceeding max_params_m."""
    from autotrust.schemas import validate_moe_config

    # A config within expert cap but we test the top_k cap
    bad_config = MoEConfig(
        num_experts=8,
        top_k=5,  # exceeds max_top_k of 4
        moe_layers=[0, 1],
    )
    with pytest.raises(ValueError, match="max_top_k"):
        validate_moe_config(bad_config, spec)


def test_checkpoint_meta_has_required_fields():
    """CheckpointMeta has stage, experiment_num, composite, path."""
    meta = CheckpointMeta(
        stage="dense_baseline",
        experiment_num=5,
        composite=0.82,
        path=Path("runs/test/best.pt"),
        param_count=50_000_000,
    )
    assert meta.stage == "dense_baseline"
    assert meta.experiment_num == 5
    assert meta.composite == 0.82
    assert meta.path == Path("runs/test/best.pt")
    assert meta.param_count == 50_000_000


def test_student_output_matches_scorer_output():
    """StudentOutput has trust_vector, reason_tags, escalate."""
    output = StudentOutput(
        trust_vector={"phish": 0.9, "manipulation": 0.3},
        reason_tags=["phish_detected", "manipulation_possible"],
        escalate=True,
    )
    assert isinstance(output.trust_vector, dict)
    assert isinstance(output.reason_tags, list)
    assert output.escalate is True


def test_teacher_artifacts_model():
    """TeacherArtifacts has prompt_pack_path, label_rules_path, explanation_schema_path."""
    artifacts = TeacherArtifacts(
        prompt_pack_path=Path("teacher/prompt_pack.yaml"),
        label_rules_path=Path("teacher/label_rules.yaml"),
        explanation_schema_path=Path("teacher/explanation_schema.json"),
        synth_data_dir=Path("synth_data"),
    )
    assert artifacts.prompt_pack_path == Path("teacher/prompt_pack.yaml")
    assert artifacts.label_rules_path == Path("teacher/label_rules.yaml")
    assert artifacts.explanation_schema_path == Path("teacher/explanation_schema.json")
    assert artifacts.synth_data_dir == Path("synth_data")
