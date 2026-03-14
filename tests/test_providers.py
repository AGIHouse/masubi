"""Tests for autotrust.providers -- registry, base classes, and concrete providers."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from autotrust.config import load_spec
from autotrust.providers import (
    get_provider,
    GeneratorProvider,
    ScoringProvider,
    JudgeProvider,
    TrainingProvider,
)


@pytest.fixture
def spec():
    return load_spec(Path(__file__).parent.parent / "spec.yaml")


# ---------------------------------------------------------------------------
# Registry tests (TASK_006)
# ---------------------------------------------------------------------------

def test_get_provider_generator(spec):
    """get_provider('generator', spec) returns a GeneratorProvider subclass."""
    provider = get_provider("generator", spec)
    assert isinstance(provider, GeneratorProvider)


def test_get_provider_scorer(spec):
    """get_provider('scorer', spec) returns a ScoringProvider subclass."""
    provider = get_provider("scorer", spec)
    assert isinstance(provider, ScoringProvider)


def test_get_provider_judge(spec):
    """get_provider('judge_primary', spec) returns a JudgeProvider subclass."""
    provider = get_provider("judge_primary", spec)
    assert isinstance(provider, JudgeProvider)


def test_get_provider_trainer(spec):
    """get_provider('trainer', spec) returns a TrainingProvider subclass."""
    provider = get_provider("trainer", spec)
    assert isinstance(provider, TrainingProvider)


def test_get_provider_unknown_role(spec):
    """get_provider('unknown', spec) raises ValueError."""
    with pytest.raises(ValueError, match="unknown"):
        get_provider("unknown", spec)


def test_base_provider_retry():
    """Retry decorator retries on transient errors up to max_retries."""
    from autotrust.providers import retry_on_error

    call_count = 0

    @retry_on_error(max_retries=3, base_delay=0.01)
    def flaky_fn():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("transient")
        return "success"

    result = flaky_fn()
    assert result == "success"
    assert call_count == 3


# ---------------------------------------------------------------------------
# Ollama provider tests (TASK_007)
# ---------------------------------------------------------------------------

class TestOllamaGenerator:
    """Tests for OllamaGenerator using mocks."""

    def test_ollama_generate_returns_string(self):
        """Mock ollama.chat, verify generate() returns string."""
        from autotrust.providers.ollama import OllamaGenerator

        gen = OllamaGenerator(model="dolphin3:latest")
        mock_response = {"message": {"content": "Hello, world!"}}
        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = mock_response
        with patch("autotrust.providers.ollama._get_ollama_client", return_value=mock_ollama):
            result = gen.generate("Say hello")
        assert isinstance(result, str)
        assert result == "Hello, world!"

    def test_ollama_generate_batch(self):
        """Mock ollama.chat, verify generate_batch() returns list with correct length."""
        from autotrust.providers.ollama import OllamaGenerator

        gen = OllamaGenerator(model="dolphin3:latest")
        mock_response = {"message": {"content": "response"}}
        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = mock_response
        with patch("autotrust.providers.ollama._get_ollama_client", return_value=mock_ollama):
            results = gen.generate_batch(["p1", "p2", "p3"])
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    def test_ollama_check_available_true(self):
        """Mock ollama.list to return model, verify check_available() returns True."""
        from autotrust.providers.ollama import OllamaGenerator

        gen = OllamaGenerator(model="dolphin3:latest")
        mock_ollama = MagicMock()
        mock_ollama.list.return_value = {"models": [{"name": "dolphin3:latest"}]}
        with patch("autotrust.providers.ollama._get_ollama_client", return_value=mock_ollama):
            assert gen.check_available() is True

    def test_ollama_check_available_false(self):
        """Mock ollama.list to raise ConnectionError, verify returns False."""
        from autotrust.providers.ollama import OllamaGenerator

        gen = OllamaGenerator(model="dolphin3:latest")
        mock_ollama = MagicMock()
        mock_ollama.list.side_effect = ConnectionError("daemon not running")
        with patch("autotrust.providers.ollama._get_ollama_client", return_value=mock_ollama):
            assert gen.check_available() is False

    def test_ollama_uses_spec_model(self, spec):
        """OllamaGenerator uses model name from spec."""
        provider = get_provider("generator", spec)
        assert provider.model == spec.providers.generator.model


# ---------------------------------------------------------------------------
# Hyperbolic provider tests (TASK_008)
# ---------------------------------------------------------------------------

class TestHyperbolicScorer:
    """Tests for HyperbolicScorer using mocks."""

    def test_hyperbolic_scorer_returns_string(self):
        """Mock openai client, verify score() returns string."""
        from autotrust.providers.hyperbolic import HyperbolicScorer

        scorer = HyperbolicScorer(model="test-model", api_key="test-key")
        mock_choice = MagicMock()
        mock_choice.message.content = "scored result"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        scorer._client = mock_client

        result = scorer.score("test prompt")
        assert isinstance(result, str)
        assert result == "scored result"

    def test_hyperbolic_scorer_batch(self):
        """Mock openai client, verify score_batch() returns list[str]."""
        from autotrust.providers.hyperbolic import HyperbolicScorer

        scorer = HyperbolicScorer(model="test-model", api_key="test-key")
        mock_choice = MagicMock()
        mock_choice.message.content = "result"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        scorer._client = mock_client

        results = scorer.score_batch(["p1", "p2"])
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    def test_hyperbolic_scorer_uses_spec_model(self, spec):
        """HyperbolicScorer uses model from spec."""
        provider = get_provider("scorer", spec)
        assert provider.model == spec.providers.scorer.model

    def test_hyperbolic_scorer_retry_on_error(self):
        """Mock client to fail then succeed, verify retry works."""
        from autotrust.providers.hyperbolic import HyperbolicScorer

        scorer = HyperbolicScorer(model="test-model", api_key="test-key")

        call_count = 0
        mock_choice = MagicMock()
        mock_choice.message.content = "success"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("transient error")
            return mock_response

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = side_effect
        scorer._client = mock_client

        result = scorer.score("test")
        assert result == "success"
        assert call_count == 2


class TestHyperbolicTrainer:
    """Tests for HyperbolicTrainer using mocks."""

    def test_hyperbolic_trainer_rent_gpu(self):
        """Mock httpx POST, verify rent_gpu returns instance_id."""
        from autotrust.providers.hyperbolic import HyperbolicTrainer

        trainer = HyperbolicTrainer(api_key="test-key", gpu_type="H100")
        mock_response = MagicMock()
        mock_response.json.return_value = {"instance_id": "inst-123"}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        trainer._client = mock_client

        result = trainer.rent_gpu(hours=2, name="test-run")
        assert result == "inst-123"

    def test_hyperbolic_trainer_stop_gpu(self):
        """Mock httpx POST, verify stop_gpu calls correct endpoint."""
        from autotrust.providers.hyperbolic import HyperbolicTrainer

        trainer = HyperbolicTrainer(api_key="test-key", gpu_type="H100")
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        trainer._client = mock_client

        trainer.stop_gpu("inst-123")
        mock_client.post.assert_called_once_with("/instances/inst-123/stop")

    def test_hyperbolic_trainer_budget_guard_triggers(self):
        """BudgetGuard raises at limit."""
        from autotrust.providers.hyperbolic import HyperbolicTrainer, BudgetExceededError

        trainer = HyperbolicTrainer(api_key="test-key", gpu_type="H100")
        guard = trainer.budget_guard(max_usd=5.0)

        with pytest.raises(BudgetExceededError):
            with guard:
                guard.track_spend(3.0)
                guard.track_spend(3.0)  # exceeds 5.0

    def test_hyperbolic_trainer_budget_guard_auto_terminates(self):
        """BudgetGuard calls stop_gpu on budget exceeded."""
        from autotrust.providers.hyperbolic import HyperbolicTrainer, BudgetExceededError

        trainer = HyperbolicTrainer(api_key="test-key", gpu_type="H100")
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        trainer._client = mock_client

        guard = trainer.budget_guard(max_usd=5.0)
        with pytest.raises(BudgetExceededError):
            with guard:
                guard.register_instance("inst-abc")
                guard.track_spend(6.0)  # exceeds immediately

        # Verify stop_gpu was called
        mock_client.post.assert_called_with("/instances/inst-abc/stop")


# ---------------------------------------------------------------------------
# Anthropic provider tests (TASK_009)
# ---------------------------------------------------------------------------

class TestAnthropicJudge:
    """Tests for AnthropicJudge using mocks."""

    def _make_judge(self):
        from autotrust.providers.anthropic import AnthropicJudge
        return AnthropicJudge(
            primary_model="claude-opus-4-20250514",
            secondary_model="claude-sonnet-4-20250514",
            api_key="test-key",
        )

    def _mock_response(self, scores_json: str):
        """Create a mock Anthropic message response."""
        mock_content = MagicMock()
        mock_content.text = scores_json
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        return mock_response

    def test_anthropic_judge_returns_per_axis_scores(self, spec):
        """Verify judge() returns dict with all spec axis names as keys."""
        judge = self._make_judge()
        axes = [a.name for a in spec.trust_axes]
        scores = {a: 0.5 for a in axes}
        import json
        mock_resp = self._mock_response(json.dumps(scores))

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp
        judge._client = mock_client

        result = judge.judge(MagicMock(emails=[]), axes)
        assert set(result.keys()) == set(axes)

    def test_anthropic_judge_scores_are_floats(self, spec):
        """All returned values are floats in [0, 1]."""
        judge = self._make_judge()
        axes = [a.name for a in spec.trust_axes]
        scores = {a: 0.7 for a in axes}
        import json
        mock_resp = self._mock_response(json.dumps(scores))

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp
        judge._client = mock_client

        result = judge.judge(MagicMock(emails=[]), axes)
        for v in result.values():
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0

    def test_anthropic_dual_judge_agreement(self, spec):
        """Both models similar scores -> agreement > 0.8."""
        judge = self._make_judge()
        axes = [a.name for a in spec.trust_axes]
        import json

        primary_scores = {a: 0.5 for a in axes}
        secondary_scores = {a: 0.5 for a in axes}  # identical

        mock_resp_p = self._mock_response(json.dumps(primary_scores))
        mock_resp_s = self._mock_response(json.dumps(secondary_scores))

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [mock_resp_p, mock_resp_s]
        judge._client = mock_client

        _, _, agreement = judge.dual_judge(MagicMock(emails=[]))
        assert agreement > 0.8

    def test_anthropic_dual_judge_disagreement(self, spec):
        """Models returning divergent scores -> disagreement detected."""
        judge = self._make_judge()
        axes = [a.name for a in spec.trust_axes]
        import json

        primary_scores = {a: 0.1 for a in axes}
        secondary_scores = {a: 0.9 for a in axes}  # very different

        mock_resp_p = self._mock_response(json.dumps(primary_scores))
        mock_resp_s = self._mock_response(json.dumps(secondary_scores))

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [mock_resp_p, mock_resp_s]
        judge._client = mock_client

        _, _, agreement = judge.dual_judge(MagicMock(emails=[]))
        assert agreement < 0.5  # high disagreement

    def test_anthropic_judge_uses_spec_models(self, spec):
        """Primary and secondary models come from spec."""
        provider = get_provider("judge_primary", spec)
        assert provider.primary_model == spec.providers.judge_primary.model
        assert provider.secondary_model == spec.providers.judge_secondary.model
