"""LLM client wrappers for text generation."""

from __future__ import annotations

import requests


class OllamaClient:
    """Client for generating text via Ollama's REST API.

    Ollama runs open-source LLMs locally and exposes an HTTP API.
    The client sends a prompt and returns the generated response.

    Args:
        model: Ollama model name (e.g. "llama3.2").
        base_url: Ollama server URL.
        temperature: Sampling temperature (0 = deterministic, higher = more random).
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The input prompt string.

        Returns:
            The generated text response.

        Raises:
            ConnectionError: If Ollama is not reachable.
            RuntimeError: If the API returns an error.
        """
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            timeout=120,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama returned status {response.status_code}: {response.text}"
            )
        return response.json()["response"]
