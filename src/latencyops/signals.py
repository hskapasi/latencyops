"""Provider-neutral normalization for model-server runtime metrics."""

from typing import Mapping

from .models import SystemSignals


class RuntimeSignalNormalizer:
    """Convert provider-specific metric names into content-free system signals."""

    def __init__(self, metric_names: Mapping[str, str] | None = None):
        self.metric_names = {
            "queue_depth": "queue_depth",
            "queue_capacity": "queue_capacity",
            "kv_cache_usage": "kv_cache_usage",
            "kv_cache_capacity": "kv_cache_capacity",
            "predicted_ttft_ms": "predicted_ttft_ms",
            "predicted_tpot_ms": "predicted_tpot_ms",
            **(metric_names or {}),
        }

    def normalize(self, metrics: Mapping[str, float], provider: str) -> SystemSignals:
        queue_depth = float(metrics.get(self.metric_names["queue_depth"], 0))
        queue_capacity = float(metrics.get(self.metric_names["queue_capacity"], 1))
        cache_used = float(metrics.get(self.metric_names["kv_cache_usage"], 0))
        cache_capacity = float(metrics.get(self.metric_names["kv_cache_capacity"], 1))
        return SystemSignals(
            queue_pressure=min(queue_depth / queue_capacity, 1.0) if queue_capacity else 1.0,
            cache_pressure=min(cache_used / cache_capacity, 1.0) if cache_capacity else 1.0,
            provider_health={provider: bool(metrics.get("healthy", 1))},
            predicted_ttft_ms={provider: float(metrics[self.metric_names["predicted_ttft_ms"]])}
            if self.metric_names["predicted_ttft_ms"] in metrics else {},
            predicted_tpot_ms={provider: float(metrics[self.metric_names["predicted_tpot_ms"]])}
            if self.metric_names["predicted_tpot_ms"] in metrics else {},
        )
