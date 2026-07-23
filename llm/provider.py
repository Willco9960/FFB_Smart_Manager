"""Provider interfaces for optional local language-model explanations.

The numerical fantasy policy remains the source of truth. Providers in this
module only turn structured evidence into a readable explanation. The default
provider is offline, so adding this package never creates an unexpected network
request during simulations or tests.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen


class CoachProvider(Protocol):
    """Minimal interface implemented by local or test coach providers."""

    def complete(self, prompt: str) -> str:
        """Return a response for a single structured coaching prompt."""


class NullCoachProvider:
    """Offline provider used when no local model is configured."""

    def complete(self, prompt: str) -> str:
        raise RuntimeError("No local coach provider is configured.")


@dataclass(frozen=True)
class LocalOpenAICompatibleProvider:
    """Call a local OpenAI-compatible chat-completions endpoint."""

    model: str
    base_url: str = "http://127.0.0.1:1234/v1"
    api_key: str = "local"
    timeout_seconds: float = 30.0

    @property
    def endpoint(self) -> str:
        normalized = self.base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        return f"{normalized}/chat/completions"

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a fantasy football assistant. "
                        "Return only the requested JSON object. "
                        "Do not invent statistics or submit transactions."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, TimeoutError) as error:
            raise RuntimeError(f"Local coach request failed: {error}") from error

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("Local coach returned an invalid chat response.") from error

        if not isinstance(content, str):
            raise RuntimeError("Local coach response content was not text.")

        return content


def create_coach_provider_from_env() -> CoachProvider:
    """Create a provider only when explicitly enabled by environment variables."""

    enabled = os.getenv("FFB_LLM_ENABLED", "0").strip().lower() in {"1", "true", "yes"}
    if not enabled:
        return NullCoachProvider()

    model = os.getenv("FFB_LLM_MODEL", "gpt-oss-20b").strip()
    base_url = os.getenv("FFB_LLM_BASE_URL", "http://127.0.0.1:1234/v1").strip()
    api_key = os.getenv("FFB_LLM_API_KEY", "local").strip()
    timeout = float(os.getenv("FFB_LLM_TIMEOUT_SECONDS", "30"))
    if not model:
        raise ValueError("FFB_LLM_MODEL must not be empty when FFB_LLM_ENABLED is enabled.")

    return LocalOpenAICompatibleProvider(
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout,
    )
