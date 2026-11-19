"""Pick the LLM provider from the LLM_PROVIDER setting."""
from app.llm.base import LLMClient
from app.llm.mock_client import MockLLMClient
from app.llm.ollama_client import OllamaClient


def create_llm_client(settings) -> LLMClient:
    if settings.llm_provider == "ollama":
        return OllamaClient(settings.ollama_base_url, settings.ollama_model,
                            settings.llm_timeout_seconds, settings.ollama_context_tokens)
    if settings.llm_provider == "mock":
        return MockLLMClient()
    raise ValueError(f"Unknown LLM_PROVIDER {settings.llm_provider!r} (use 'ollama' or 'mock')")
