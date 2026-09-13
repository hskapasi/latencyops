"""Provider adapters built on the standard library HTTP client."""

import json
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .capabilities import ProviderCapabilities
from .models import InferencePlan, StreamChunk


class ProviderError(RuntimeError):
    """Normalized provider failure without including request content."""


@dataclass(frozen=True)
class ProviderHealth:
    """Result of a provider health probe."""

    healthy: bool
    detail: str


class CallableProvider:
    """Wrap a local callable as a deterministic inference provider."""

    def __init__(self, name: str, responder: Callable[[str, InferencePlan], str]):
        self.name = name
        self.responder = responder

    def generate(self, prompt: str, plan: InferencePlan) -> str:
        return self.responder(prompt, plan)

    def stream(self, prompt: str, plan: InferencePlan) -> Iterable[StreamChunk]:
        yield StreamChunk(self.generate(prompt, plan))


class OpenAICompatibleProvider:
    """Minimal adapter for JSON-compatible OpenAI-style completion endpoints."""

    def __init__(
        self,
        name: str,
        endpoint: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        model_names: Mapping[str, str] | None = None,
        health_endpoint: str | None = None,
    ):
        self.name = name
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.model_names = dict(model_names or {})
        self.health_endpoint = health_endpoint
        self.capabilities = ProviderCapabilities()

    def _model_name(self, plan: InferencePlan) -> str:
        return self.model_names.get(plan.model_tier, plan.model_tier)

    def _headers(self, streaming: bool = False) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if streaming:
            headers["Accept"] = "text/event-stream"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate(self, prompt: str, plan: InferencePlan) -> str:
        payload = json.dumps({"model": self._model_name(plan), "prompt": prompt, "stream": False}).encode()
        request = Request(self.endpoint, data=payload, headers=self._headers(), method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
        except (HTTPError, URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            raise ProviderError(f"provider {self.name!r} request failed") from exc
        choices = body.get("choices", [])
        if not choices or "text" not in choices[0]:
            raise ProviderError(f"provider {self.name!r} returned an invalid response")
        return choices[0]["text"]

    def stream(self, prompt: str, plan: InferencePlan) -> Iterable[StreamChunk]:
        payload = json.dumps({"model": self._model_name(plan), "prompt": prompt, "stream": True}).encode()
        request = Request(self.endpoint, data=payload, headers=self._headers(streaming=True), method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode().strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    text = event.get("choices", [{}])[0].get("text", "")
                    if text:
                        yield StreamChunk(text)
        except (HTTPError, URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            raise ProviderError(f"provider {self.name!r} stream failed") from exc

    def health(self) -> ProviderHealth:
        if not self.health_endpoint:
            return ProviderHealth(True, "health probe not configured")
        request = Request(self.health_endpoint, headers=self._headers(), method="GET")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return ProviderHealth(200 <= response.status < 300, "HTTP health probe completed")
        except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            return ProviderHealth(False, "HTTP health probe failed")


class OpenAIChatCompatibleProvider(OpenAICompatibleProvider):
    """Adapter for OpenAI-compatible chat-completion endpoints."""

    def __init__(self, *args, chat_template_kwargs: Mapping[str, object] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat_template_kwargs = dict(chat_template_kwargs or {})

    def _payload(self, prompt: str, plan: InferencePlan, streaming: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self._model_name(plan),
            "messages": [{"role": "user", "content": prompt}],
            "stream": streaming,
        }
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        return payload

    def generate(self, prompt: str, plan: InferencePlan) -> str:
        request = Request(
            self.endpoint,
            data=json.dumps(self._payload(prompt, plan, False)).encode(),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
        except (HTTPError, URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            raise ProviderError(f"provider {self.name!r} request failed") from exc
        choices = body.get("choices", [])
        content = choices[0].get("message", {}).get("content") if choices else None
        if not isinstance(content, str):
            raise ProviderError(f"provider {self.name!r} returned an invalid chat response")
        return content

    def stream(self, prompt: str, plan: InferencePlan) -> Iterable[StreamChunk]:
        request = Request(
            self.endpoint,
            data=json.dumps(self._payload(prompt, plan, True)).encode(),
            headers=self._headers(streaming=True),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode().strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    choice = event.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        yield StreamChunk(text)
        except (HTTPError, URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            raise ProviderError(f"provider {self.name!r} chat stream failed") from exc

    def list_models(self) -> list[str]:
        """Return model IDs visible to this provider credential."""
        endpoint = self.endpoint.rsplit("/v1/", 1)[0] + "/v1/models"
        request = Request(endpoint, headers=self._headers(), method="GET")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
        except (HTTPError, URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            raise ProviderError(f"provider {self.name!r} model listing failed") from exc
        return [item["id"] for item in body.get("data", []) if isinstance(item.get("id"), str)]


class VLLMCompatibleProvider(OpenAIChatCompatibleProvider):
    """vLLM adapter using its OpenAI-compatible chat-completions endpoint."""
