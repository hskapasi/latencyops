"""Local OpenAI-compatible gateway for the LatencyOps operational core."""

import json
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from .benchmark import BenchmarkRunner
from .capabilities import ProviderCapabilities, enforce_plan
from .controller import ProactiveController
from .models import LatencyContract, ProviderCandidate, RequestProfile, SystemSignals
from .policy import AdaptiveProviderSelector
from .router import InferenceProvider, ModelRouter
from .scheduling import CachePressureSignal, PriorityScheduler
from .telemetry import TelemetryRecord, TelemetryRecorder


class GatewayService:
    """Translate completion requests into planned provider calls."""

    def __init__(
        self,
        router: ModelRouter,
        planner,
        telemetry: TelemetryRecorder | None = None,
        provider_candidates: list[ProviderCandidate] | None = None,
        provider_lookup: dict[str, InferenceProvider] | None = None,
    ):
        self.router = router
        self.planner = planner
        self.controller = ProactiveController(planner)
        self.telemetry = telemetry or TelemetryRecorder()
        self.provider_candidates = provider_candidates or []
        self.provider_lookup = provider_lookup or {}
        self.provider_selector = AdaptiveProviderSelector()
        self.scheduler = PriorityScheduler[dict[str, Any]]()
        self.benchmark = BenchmarkRunner()

    def _plan_request(self, body: dict[str, Any]):
        prompt = body.get("prompt", "")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        contract = LatencyContract(
            deadline_ms=float(body.get("deadline_ms", 1000)),
            quality_floor=float(body.get("quality_floor", 0.9)),
            risk_class=body.get("risk_class", "medium"),
        )
        cache = CachePressureSignal(int(body.get("cache_used", 0)), int(body.get("cache_capacity", 1)))
        profile = RequestProfile(
            prompt_tokens=len(prompt.split()),
            expected_output_tokens=int(body.get("max_tokens", 16)),
            difficulty=float(body.get("difficulty", 0.5)),
            cache_pressure=cache.pressure,
            queue_pressure=min(self.scheduler.depth / 100, 1.0),
        )
        signals = SystemSignals(
            queue_pressure=min(self.scheduler.depth / 100, 1.0),
            cache_pressure=cache.pressure,
        )
        return prompt, contract, self.controller.initial_plan(contract, profile, signals), cache.pressure

    def _provider_for(self, contract: LatencyContract, plan):
        if self.provider_candidates:
            candidates = [candidate for candidate in self.provider_candidates if candidate.model_tier == plan.model_tier]
            selected = self.provider_selector.select(contract, candidates)
            provider = self.provider_lookup[selected.name]
            return provider, selected.name
        return self.router.providers[plan.model_tier], self.router.providers[plan.model_tier].name

    def complete(self, body: dict[str, Any]) -> dict[str, Any]:
        prompt, contract, plan, cache_pressure = self._plan_request(body)
        provider, provider_name = self._provider_for(contract, plan)
        capabilities = getattr(provider, "capabilities", ProviderCapabilities())
        enforcement = enforce_plan(contract, plan, capabilities)
        plan = enforcement.enforced
        result = self.benchmark.run_streaming(provider, prompt, plan)
        self.telemetry.record(TelemetryRecord(provider_name, plan.model_tier, result.sample, cache_pressure=cache_pressure))
        return {
            "id": "latencyops-local",
            "object": "text_completion",
            "choices": [{"text": result.text, "index": 0, "finish_reason": "stop"}],
            "latencyops": {
                "plan": asdict(enforcement.enforced),
                "plan_requested": asdict(enforcement.requested),
                "plan_enforced": asdict(enforcement.enforced),
                "unsupported_features": list(enforcement.unsupported),
                "ttft_ms": result.sample.ttft_ms,
                "tpot_ms": result.sample.tpot_ms,
                "end_to_end_ms": result.sample.end_to_end_ms,
            },
        }

    def stream(self, body: dict[str, Any]):
        """Yield provider-neutral chunks for an HTTP streaming response."""
        prompt, contract, plan, _ = self._plan_request(body)
        provider, _ = self._provider_for(contract, plan)
        capabilities = getattr(provider, "capabilities", ProviderCapabilities())
        return provider.stream(prompt, enforce_plan(contract, plan, capabilities).enforced)


def create_server(host: str, port: int, service: GatewayService) -> ThreadingHTTPServer:
    """Create a local HTTP server; call ``serve_forever`` to run it."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/metrics":
                exporter = next((item for item in service.telemetry.exporters if hasattr(item, "render")), None)
                payload = exporter.render() if exporter else ""
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.end_headers()
                self.wfile.write(payload.encode())
                return
            self.send_error(404)

        def do_POST(self):
            if self.path != "/v1/completions":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                if body.get("stream", False):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    for chunk in service.stream(body):
                        event = {"choices": [{"text": chunk.text, "index": 0, "finish_reason": None}]}
                        self.wfile.write(f"data: {json.dumps(event)}\\n\\n".encode())
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\\n\\n")
                    return
                result = service.complete(body)
                payload = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.send_error(400, str(exc))
            except Exception as exc:
                self.send_error(502, str(exc))

        def log_message(self, format, *args):
            return

    return ThreadingHTTPServer((host, port), Handler)
