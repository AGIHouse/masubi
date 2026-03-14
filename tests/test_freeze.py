"""Tests for autotrust.freeze -- Teacher artifact extraction."""

import json
import pytest
import yaml
from pathlib import Path

from autotrust.config import load_spec
from autotrust.schemas import TeacherArtifacts


@pytest.fixture
def spec():
    return load_spec(Path(__file__).parent.parent / "spec.yaml")


@pytest.fixture
def sample_train_py():
    """Sample train.py source code for extraction tests."""
    return Path(__file__).parent.parent / "train.py"


@pytest.fixture
def train_py_source(sample_train_py):
    return sample_train_py.read_text()


def test_extract_prompt_pack(train_py_source):
    """Given a train.py string, extracts prompt template into YAML structure."""
    from autotrust.freeze import extract_prompt_pack
    pack = extract_prompt_pack(train_py_source)
    assert isinstance(pack, dict)
    assert "prompt_template" in pack
    assert len(pack["prompt_template"]) > 0


def test_extract_label_rules(train_py_source):
    """Extracts threshold/escalation rules from train.py code."""
    from autotrust.freeze import extract_label_rules
    rules = extract_label_rules(train_py_source)
    assert isinstance(rules, dict)
    assert "flag_threshold" in rules


def test_extract_explanation_schema(train_py_source, spec):
    """Extracts reason tag vocabulary."""
    from autotrust.freeze import extract_explanation_schema
    schema = extract_explanation_schema(train_py_source, spec)
    assert isinstance(schema, dict)
    assert "axis_names" in schema
    assert len(schema["axis_names"]) == len(spec.trust_axes)


def test_write_teacher_artifacts(tmp_path, train_py_source, spec):
    """Writes prompt_pack.yaml, label_rules.yaml, explanation_schema.json to teacher/."""
    from autotrust.freeze import write_teacher_artifacts
    artifacts = write_teacher_artifacts(train_py_source, spec, tmp_path / "teacher")
    assert (tmp_path / "teacher" / "prompt_pack.yaml").exists()
    assert (tmp_path / "teacher" / "label_rules.yaml").exists()
    assert (tmp_path / "teacher" / "explanation_schema.json").exists()


def test_freeze_creates_teacher_dir(tmp_path, train_py_source, spec):
    """teacher/ directory is created if it does not exist."""
    from autotrust.freeze import write_teacher_artifacts
    teacher_dir = tmp_path / "teacher"
    assert not teacher_dir.exists()
    write_teacher_artifacts(train_py_source, spec, teacher_dir)
    assert teacher_dir.exists()
    assert teacher_dir.is_dir()


def test_freeze_returns_teacher_artifacts(tmp_path, train_py_source, spec):
    """Returns a TeacherArtifacts model with correct paths."""
    from autotrust.freeze import write_teacher_artifacts
    teacher_dir = tmp_path / "teacher"
    artifacts = write_teacher_artifacts(train_py_source, spec, teacher_dir)
    assert isinstance(artifacts, TeacherArtifacts)
    assert artifacts.prompt_pack_path == teacher_dir / "prompt_pack.yaml"
    assert artifacts.label_rules_path == teacher_dir / "label_rules.yaml"
    assert artifacts.explanation_schema_path == teacher_dir / "explanation_schema.json"


def test_freeze_from_git_history(monkeypatch, tmp_path, spec):
    """Given a mocked git log with composite scores, selects the best commit and extracts artifacts."""
    from autotrust.freeze import freeze_teacher

    # Mock git operations
    mock_log = [
        {"hash": "abc123", "message": "composite=0.72", "date": "2024-01-01", "composite": 0.72},
        {"hash": "def456", "message": "composite=0.85", "date": "2024-01-02", "composite": 0.85},
        {"hash": "ghi789", "message": "composite=0.60", "date": "2024-01-03", "composite": 0.60},
    ]

    # Read real train.py source for extraction
    real_source = (Path(__file__).parent.parent / "train.py").read_text()

    import autotrust.freeze as freeze_mod
    monkeypatch.setattr(freeze_mod, "_get_train_py_log", lambda: mock_log)
    monkeypatch.setattr(freeze_mod, "_get_file_at_commit", lambda h, f="train.py": real_source)

    artifacts = freeze_teacher(spec, teacher_dir=tmp_path / "teacher")
    assert isinstance(artifacts, TeacherArtifacts)
    assert (tmp_path / "teacher" / "prompt_pack.yaml").exists()


def test_relabel_training_data(tmp_path, train_py_source, spec):
    """relabel_training_data writes updated JSONL with soft trust vectors."""
    from autotrust.freeze import write_teacher_artifacts, relabel_training_data

    teacher_dir = tmp_path / "teacher"
    synth_dir = tmp_path / "synth_data"
    synth_dir.mkdir(parents=True)

    # Create a minimal train.jsonl with existing labels
    records = [
        {
            "chain_id": "c1",
            "emails": [
                {
                    "from_addr": "a@b.com", "to_addr": "c@d.com",
                    "subject": "Test", "body": "Hello",
                    "timestamp": "2024-01-01T00:00:00", "reply_depth": 0,
                }
            ],
            "labels": {"phish": 0.8, "truthfulness": 0.5},
            "trust_vector": {"phish": 0.8, "truthfulness": 0.5},
            "composite": 0.65,
            "flags": ["phish"],
        }
    ]
    with open(synth_dir / "train.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # Write teacher artifacts
    artifacts = write_teacher_artifacts(train_py_source, spec, teacher_dir)
    # Override synth_data_dir to our tmp location
    artifacts = TeacherArtifacts(
        prompt_pack_path=artifacts.prompt_pack_path,
        label_rules_path=artifacts.label_rules_path,
        explanation_schema_path=artifacts.explanation_schema_path,
        synth_data_dir=synth_dir,
    )

    # Call relabel (will use fallback path since no API provider)
    output_path = relabel_training_data(artifacts, spec)
    assert output_path.exists()

    # Verify output has soft_targets
    output_lines = output_path.read_text().strip().split("\n")
    assert len(output_lines) == 1
    output_record = json.loads(output_lines[0])
    assert "soft_targets" in output_record
