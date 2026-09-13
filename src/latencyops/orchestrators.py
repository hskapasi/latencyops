"""HTTP/Prometheus collectors for confirmed model-serving interfaces."""

import re
from typing import Mapping
from urllib.request import Request, urlopen

from .models import SystemSignals
from .signals import RuntimeSignalNormalizer


PROFILES: dict[str, dict[str, str]] = {
    "vllm": {
        "queue_depth": "vllm:num_requests_waiting",
        "cache_pressure": "vllm:kv_cache_usage_perc",
        "ttft": "vllm:time_to_first_token_seconds",
        "tpot": "vllm:inter_token_latency_seconds",
    },
    "sglang": {
        "queue_depth": "sglang:num_queue_reqs",
        "cache_pressure": "sglang:token_usage",
        "cache_hit_rate": "sglang:cache_hit_rate",
    },
    "tensorrtllm": {
        "queue_depth": "nv_trt_llm_request_metrics",
        "cache_pressure": "nv_trt_llm_kv_cache_block_metrics",
    },
    "llm-d": {
        "queue_depth": "llm_d_epp_",
    },
    "litellm": {
        "requests": "litellm_proxy_total_requests_metric",
        "failures": "litellm_proxy_failed_requests_metric",
    },
}


class MetricsEndpointError(RuntimeError):
    """Raised when a metrics endpoint cannot be fetched or parsed."""


class PrometheusMetricsCollector:
    """Fetch a Prometheus endpoint and normalize selected gauges."""

    def __init__(
        self,
        endpoint: str,
        provider: str,
        api_key: str | None = None,
        queue_capacity: float = 100.0,
        metric_names: Mapping[str, str] | None = None,
        timeout: float = 10.0,
    ):
        self.endpoint = endpoint
        self.provider = provider
        self.api_key = api_key
        self.queue_capacity = queue_capacity
        self.metric_names = {**PROFILES.get(provider, {}), **(metric_names or {})}
        self.timeout = timeout

    def fetch(self) -> SystemSignals:
        headers = {"Accept": "text/plain"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            with urlopen(Request(self.endpoint, headers=headers), timeout=self.timeout) as response:
                payload = response.read().decode()
        except Exception as exc:
            raise MetricsEndpointError(f"could not fetch {self.provider} metrics") from exc
        metrics = self.parse(payload)
        return self.normalize(metrics)

    def parse(self, payload: str) -> dict[str, float]:
        """Parse simple gauge lines; histogram buckets are intentionally ignored."""
        values: dict[str, float] = {}
        for line in payload.splitlines():
            if not line or line.startswith("#"):
                continue
            match = re.match(r"([^\s{]+)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)$", line)
            if match:
                name, labels, value = match.groups()
                labels = labels or ""
                if name == "nv_trt_llm_request_metrics" and 'request_type="waiting"' not in labels:
                    continue
                if name == "nv_trt_llm_kv_cache_block_metrics" and 'kv_cache_block_type="fraction"' not in labels:
                    continue
                try:
                    values[name] = float(value)
                except ValueError:
                    continue
        return values

    def normalize(self, metrics: Mapping[str, float]) -> SystemSignals:
        queue_name = self.metric_names.get("queue_depth")
        cache_name = self.metric_names.get("cache_pressure")
        queue_depth = sum(value for name, value in metrics.items() if queue_name and name.startswith(queue_name))
        cache_value = metrics.get(cache_name, 0.0) if cache_name else 0.0
        if self.provider == "vllm" and cache_value > 1:
            cache_value /= 100
        return RuntimeSignalNormalizer({
            "queue_depth": "__queue_depth",
            "queue_capacity": "__queue_capacity",
            "kv_cache_usage": "__cache_pressure",
            "kv_cache_capacity": "__cache_capacity",
        }).normalize({
            "__queue_depth": queue_depth,
            "__queue_capacity": self.queue_capacity,
            "__cache_pressure": min(max(cache_value, 0.0), 1.0),
            "__cache_capacity": 1.0,
            "healthy": 1,
        }, self.provider)


class VLLMMetricsCollector(PrometheusMetricsCollector):
    """Collector for vLLM's documented `/metrics` endpoint."""

    def __init__(self, endpoint: str, **kwargs):
        super().__init__(endpoint, "vllm", **kwargs)


class SGLangMetricsCollector(PrometheusMetricsCollector):
    """Collector for SGLang metrics enabled with `--enable-metrics`."""

    def __init__(self, endpoint: str, **kwargs):
        super().__init__(endpoint, "sglang", **kwargs)


class TensorRTLLMMetricsCollector(PrometheusMetricsCollector):
    """Collector for TensorRT-LLM/Triton KV-cache and queue metrics."""

    def __init__(self, endpoint: str, **kwargs):
        super().__init__(endpoint, "tensorrtllm", **kwargs)
