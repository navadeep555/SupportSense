"""Tests for the Ollama HTTP client using a fake HTTP server (no Ollama needed)."""
import json

import httpx2
import pytest

from app.llm.base import LLMResponseError, LLMTimeoutError, LLMUnavailableError
from app.llm.ollama_client import OllamaClient


def client_with(handler) -> OllamaClient:
    return OllamaClient("http://ollama.test", "qwen2.5:7b", timeout_seconds=5,
                        transport=httpx2.MockTransport(handler))


def test_builds_request_and_parses_reply():
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json={
            "model": "qwen2.5:7b", "message": {"role": "assistant", "content": '{"answer": "hi"}'},
            "done": True, "prompt_eval_count": 812, "eval_count": 95})

    result = client_with(handler).generate("RULES", "TICKET", {"type": "object"})

    assert seen["path"] == "/api/chat"
    body = seen["body"]
    assert body["messages"] == [{"role": "system", "content": "RULES"}, {"role": "user", "content": "TICKET"}]
    assert body["stream"] is False
    assert body["format"] == {"type": "object"}
    assert body["options"]["temperature"] == 0 and body["options"]["num_ctx"] == 8192
    assert result.content == '{"answer": "hi"}'
    assert (result.input_tokens, result.output_tokens) == (812, 95)


def test_connection_refused_becomes_unavailable_error():
    def handler(request):
        raise httpx2.ConnectError("refused", request=request)

    with pytest.raises(LLMUnavailableError, match="Is Ollama installed and running"):
        client_with(handler).generate("s", "u", {})


def test_timeout_becomes_timeout_error():
    def handler(request):
        raise httpx2.ReadTimeout("slow", request=request)

    with pytest.raises(LLMTimeoutError):
        client_with(handler).generate("s", "u", {})


def test_missing_model_gives_pull_instruction():
    client = client_with(lambda r: httpx2.Response(404, json={"error": "model not found"}))
    with pytest.raises(LLMResponseError, match="ollama pull qwen2.5:7b"):
        client.generate("s", "u", {})


def test_malformed_body_becomes_response_error():
    client = client_with(lambda r: httpx2.Response(200, json={"unexpected": True}))
    with pytest.raises(LLMResponseError, match="unexpected response"):
        client.generate("s", "u", {})


def test_is_available_checks_model_is_downloaded():
    tags = {"models": [{"name": "qwen2.5:7b"}, {"name": "llama3.1:latest"}]}
    assert client_with(lambda r: httpx2.Response(200, json=tags)).is_available()
    other = OllamaClient("http://ollama.test", "mistral", 5,
                         transport=httpx2.MockTransport(lambda r: httpx2.Response(200, json=tags)))
    assert not other.is_available()


def test_is_available_false_when_server_down():
    def handler(request):
        raise httpx2.ConnectError("refused", request=request)

    assert not client_with(handler).is_available()
