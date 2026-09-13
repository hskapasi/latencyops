"""Run the local gateway with synthetic providers and no credentials."""

from latencyops.adapters import CallableProvider
from latencyops.gateway import GatewayService, create_server
from latencyops.policy import ElasticInferencePlanner
from latencyops.router import ModelRouter
from latencyops.telemetry import PrometheusExporter, TelemetryRecorder


providers = {
    tier: CallableProvider(tier, lambda prompt, plan, tier=tier: f"{tier}: synthetic response")
    for tier in ("small", "standard", "large")
}
exporter = PrometheusExporter()
service = GatewayService(ModelRouter(providers), ElasticInferencePlanner(), TelemetryRecorder([exporter]))

if __name__ == "__main__":
    print("LatencyOps synthetic gateway listening on http://127.0.0.1:8080")
    create_server("127.0.0.1", 8080, service).serve_forever()
