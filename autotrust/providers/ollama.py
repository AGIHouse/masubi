"""OllamaGenerator -- GeneratorProvider for local LLM text generation via Ollama."""

from __future__ import annotations

from typing import Any

from autotrust.providers import GeneratorProvider, retry_on_error


def _get_ollama_client():
    """Lazy import of ollama package."""
    import ollama
    return ollama


class OllamaGenerator(GeneratorProvider):
    """Generate text using the local Ollama daemon."""

    def __init__(self, model: str) -> None:
        self.model = model

    @retry_on_error(max_retries=3, base_delay=0.5)
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from a prompt via ollama.chat()."""
        self._log_call("generate", prompt_len=len(prompt))
        client = _get_ollama_client()
        response = client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

    def generate_batch(self, prompts: list[str], concurrency: int = 4) -> list[str]:
        """Generate text for multiple prompts (sequential for now)."""
        return [self.generate(p) for p in prompts]

    def check_available(self) -> bool:
        """Check if Ollama daemon is running and model is pulled."""
        try:
            client = _get_ollama_client()
            models = client.list()
            model_names = [m.get("name", "") for m in models.get("models", [])]
            return any(self.model in name for name in model_names)
        except Exception:
            return False
