"""TOML configuration for the installable LatencyOps application."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .adapters import OpenAIChatCompatibleProvider
from .gateway import GatewayService
from .models import ProviderCandidate
from .policy import ElasticInferencePlanner
from .router import ModelRouter
from .telemetry import PrometheusExporter, TelemetryRecorder


@dataclass(frozen=True)
class AppConfig:
    """Validated runtime configuration loaded from TOML and environment variables."""

    host: str
    port: int
    providers: dict[str, OpenAIChatCompatibleProvider]
    candidates: list[ProviderCandidate]


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    server = raw.get("server", {})
    provider_table = raw.get("providers", {})
    providers: dict[str, OpenAIChatCompatibleProvider] = {}
    candidates: list[ProviderCandidate] = []
    for name, settings in provider_table.items():
        endpoint = settings.get("endpoint")
        key_env = settings.get("api_key_env")
        if not endpoint or not key_env:
            raise ValueError(f"provider {name!r} requires endpoint and api_key_env")
        key = os.environ.get(key_env)
        if not key:
            raise ValueError(f"environment variable {key_env!r} is not set")
        models = settings.get("models", {})
        provider = OpenAIChatCompatibleProvider(
            name,
            endpoint,
            api_key=key,
            timeout=float(settings.get("timeout_seconds", 30)),
            model_names=models,
            chat_template_kwargs=settings.get("chat_template_kwargs"),
            health_endpoint=settings.get("health_endpoint"),
        )
        providers[name] = provider
        for tier, model in models.items():
            candidates.append(ProviderCandidate(name, tier, True, cost_per_1k_tokens=settings.get("cost_per_1k_tokens")))
    missing = {"small", "standard", "large"} - {candidate.model_tier for candidate in candidates}
    if missing:
        raise ValueError(f"configuration is missing model tiers: {sorted(missing)}")
    return AppConfig(str(server.get("host", "127.0.0.1")), int(server.get("port", 8080)), providers, candidates)


def build_service(config: AppConfig) -> GatewayService:
    exporter = PrometheusExporter()
    telemetry = TelemetryRecorder([exporter])
    lookup = {candidate.name: config.providers[candidate.name] for candidate in config.candidates}
    by_tier = {
        candidate.model_tier: config.providers[candidate.name]
        for candidate in config.candidates
    }
    return GatewayService(
        ModelRouter(by_tier), ElasticInferencePlanner(), telemetry,
        provider_candidates=config.candidates, provider_lookup=lookup,
    )
