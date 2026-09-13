"""Provider-neutral model routing contracts."""

from dataclasses import dataclass
from typing import Iterable, Protocol

from .models import InferencePlan, StreamChunk


class InferenceProvider(Protocol):
    name: str

    def generate(self, prompt: str, plan: InferencePlan) -> str:
        ...


class StreamingProvider(Protocol):
    name: str

    def stream(self, prompt: str, plan: InferencePlan) -> Iterable[StreamChunk]:
        ...


@dataclass
class StaticProvider:
    """Small local provider useful for examples and deterministic tests."""

    name: str
    response: str = ""

    def generate(self, prompt: str, plan: InferencePlan) -> str:
        del prompt, plan
        return self.response

    def stream(self, prompt: str, plan: InferencePlan) -> Iterable[StreamChunk]:
        del prompt, plan
        if self.response:
            yield StreamChunk(self.response)


class ModelRouter:
    """Route plans to providers by model tier."""

    def __init__(self, providers: dict[str, InferenceProvider]):
        required = {"small", "standard", "large"}
        missing = required - providers.keys()
        if missing:
            raise ValueError(f"missing providers: {sorted(missing)}")
        self.providers = providers

    def generate(self, prompt: str, plan: InferencePlan) -> str:
        return self.providers[plan.model_tier].generate(prompt, plan)

    def stream(self, prompt: str, plan: InferencePlan) -> Iterable[StreamChunk]:
        provider = self.providers[plan.model_tier]
        stream = getattr(provider, "stream", None)
        if not callable(stream):
            raise TypeError(f"provider {provider.name!r} does not support streaming")
        return stream(prompt, plan)
