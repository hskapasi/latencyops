"""Dependency-light telemetry records and exporter interfaces."""

from dataclasses import dataclass
from typing import Protocol

from .models import LatencySample


@dataclass(frozen=True)
class TelemetryRecord:
    """Aggregated request telemetry without prompt or response content."""

    provider: str
    model_tier: str
    sample: LatencySample
    queue_ms: float = 0.0
    cache_pressure: float = 0.0


class TelemetryExporter(Protocol):
    def export(self, record: TelemetryRecord) -> None:
        ...


class TelemetryRecorder:
    """Record bounded in-process telemetry and fan it out to exporters."""

    def __init__(self, exporters: list[TelemetryExporter] | None = None):
        self.records: list[TelemetryRecord] = []
        self.exporters = exporters or []

    def record(self, record: TelemetryRecord) -> None:
        self.records.append(record)
        for exporter in self.exporters:
            exporter.export(record)


class PrometheusExporter:
    """Render request telemetry as a Prometheus text exposition snapshot."""

    def __init__(self):
        self.records: list[TelemetryRecord] = []

    def export(self, record: TelemetryRecord) -> None:
        self.records.append(record)

    def render(self) -> str:
        lines = [
            "# HELP latencyops_requests_total Inference requests observed",
            "# TYPE latencyops_requests_total counter",
            f"latencyops_requests_total {len(self.records)}",
        ]
        if self.records:
            successful = sum(record.sample.success for record in self.records)
            lines.extend([
                "# HELP latencyops_ttft_ms_last Last observed time to first token",
                "# TYPE latencyops_ttft_ms_last gauge",
                f"latencyops_ttft_ms_last {self.records[-1].sample.ttft_ms}",
                "# HELP latencyops_success_rate Request success ratio",
                "# TYPE latencyops_success_rate gauge",
                f"latencyops_success_rate {successful / len(self.records)}",
            ])
        return "\n".join(lines) + "\n"


class OpenTelemetryExporter:
    """Adapter hook for an OpenTelemetry-compatible callback."""

    def __init__(self, emit):
        self.emit = emit

    def export(self, record: TelemetryRecord) -> None:
        self.emit({
            "provider": record.provider,
            "model_tier": record.model_tier,
            "ttft_ms": record.sample.ttft_ms,
            "tpot_ms": record.sample.tpot_ms,
            "end_to_end_ms": record.sample.end_to_end_ms,
            "queue_ms": record.queue_ms,
            "cache_pressure": record.cache_pressure,
            "success": record.sample.success,
        })
