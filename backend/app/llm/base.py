"""The contract every LLM provider follows, plus the errors they may raise.

The rest of SupportSense only talks to this interface, never to Ollama (or any
other vendor) directly. Swapping providers = writing one new class.
"""
from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResult:
    content: str                # raw text the model returned (we ask for JSON)
    provider: str               # "ollama", "mock", ...
    model: str
    input_tokens: int | None    # prompt size in tokens, if the provider reports it
    output_tokens: int | None   # generated tokens, if reported
    latency_ms: int             # wall-clock time of the call


class LLMError(Exception):
    """Base class: any failure to get a usable reply from the LLM."""


class LLMUnavailableError(LLMError):
    """Server not reachable (not installed, not running, wrong URL)."""


class LLMTimeoutError(LLMError):
    """Server reachable but did not answer in time."""


class LLMResponseError(LLMError):
    """Server answered with an error status or a malformed body."""


class LLMClient(Protocol):
    provider: str
    model: str

    def generate(self, system: str, user: str, json_schema: dict) -> LLMResult:
        """Send system instructions + user content; ask for JSON matching json_schema."""
        ...

    def is_available(self) -> bool:
        """Cheap check used by /api/health."""
        ...
