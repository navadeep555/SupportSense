"""Ollama provider: calls a local Ollama server over its REST API.

Ollama runs open models (here qwen2.5:7b) on your own machine at
http://localhost:11434. Free, offline, and ticket text never leaves the laptop.

We call POST /api/chat directly with an HTTP client instead of a wrapper
library, so every field sent and received is visible here.
"""
import time

import httpx2

from app.llm.base import LLMResponseError, LLMResult, LLMTimeoutError, LLMUnavailableError


class OllamaClient:
    provider = "ollama"

    def __init__(self, base_url: str, model: str, timeout_seconds: float,
                 context_tokens: int = 8192, transport: httpx2.BaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.context_tokens = context_tokens
        # connect=5: fail fast if Ollama isn't running; the overall timeout is
        # long because the first call also loads the model into memory.
        # `transport` lets tests plug in a fake server (httpx2.MockTransport).
        self._http = httpx2.Client(base_url=self.base_url,
                                   timeout=httpx2.Timeout(timeout_seconds, connect=5.0),
                                   transport=transport)

    def generate(self, system: str, user: str, json_schema: dict) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},  # our rules
                {"role": "user", "content": user},      # ticket + retrieved sources
            ],
            "stream": False,          # one complete JSON reply instead of token-by-token
            "format": json_schema,    # structured output: constrain reply to this JSON schema
            "options": {
                "temperature": 0,     # least random: same input -> (almost) same draft
                # Context window. If the prompt were longer than this, Ollama would
                # silently drop the beginning, so set it explicitly.
                "num_ctx": self.context_tokens,
            },
        }
        started = time.perf_counter()
        try:
            response = self._http.post("/api/chat", json=payload)
        except httpx2.ConnectError as exc:
            raise LLMUnavailableError(
                f"Cannot reach Ollama at {self.base_url}. Is Ollama installed and running?") from exc
        except httpx2.TimeoutException as exc:
            raise LLMTimeoutError(f"Ollama did not answer within the timeout ({self.model}).") from exc
        except httpx2.RequestError as exc:
            raise LLMUnavailableError(f"Network error talking to Ollama: {exc}") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        if response.status_code == 404:
            raise LLMResponseError(f"Model {self.model!r} not found in Ollama. Run: ollama pull {self.model}")
        if response.status_code >= 400:
            raise LLMResponseError(f"Ollama returned HTTP {response.status_code}: {response.text[:200]}")

        try:
            data = response.json()
            content = data["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMResponseError("Ollama returned an unexpected response body.") from exc

        return LLMResult(
            content=content,
            provider=self.provider,
            model=data.get("model", self.model),
            input_tokens=data.get("prompt_eval_count"),  # Ollama's name for prompt tokens
            output_tokens=data.get("eval_count"),        # Ollama's name for generated tokens
            latency_ms=latency_ms,
        )

    def is_available(self) -> bool:
        """True if the server answers AND the configured model is downloaded."""
        try:
            response = self._http.get("/api/tags", timeout=2.0)  # lists local models
            names = {m["name"] for m in response.json().get("models", [])}
        except (httpx2.RequestError, ValueError, KeyError, TypeError):
            return False
        return self.model in names or f"{self.model}:latest" in names
