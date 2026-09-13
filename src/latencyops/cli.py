"""Command-line entry point for the LatencyOps reference application."""

import argparse
import json
from dataclasses import asdict

from .config import build_service, load_config
from .models import LatencyContract, RequestProfile, SystemSignals
from .policy import ElasticInferencePlanner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="latencyops")
    subparsers = parser.add_subparsers(dest="command", required=True)
    gateway = subparsers.add_parser("gateway", help="start the configured HTTP gateway")
    gateway.add_argument("--config", required=True)
    plan = subparsers.add_parser("plan", help="calculate a plan from a JSON request")
    plan.add_argument("--request", required=True)
    health = subparsers.add_parser("health", help="check configured provider health")
    health.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        with open(args.request, encoding="utf-8") as handle:
            request = json.load(handle)
        contract = LatencyContract(
            deadline_ms=float(request.get("deadline_ms", 1000)),
            quality_floor=float(request.get("quality_floor", 0.9)),
            risk_class=request.get("risk_class", "medium"),
        )
        profile = RequestProfile(
            prompt_tokens=int(request.get("prompt_tokens", 0)),
            expected_output_tokens=int(request.get("expected_output_tokens", 16)),
            difficulty=float(request.get("difficulty", 0.5)),
            cache_pressure=float(request.get("cache_pressure", 0)),
            queue_pressure=float(request.get("queue_pressure", 0)),
        )
        print(json.dumps(asdict(ElasticInferencePlanner().plan(contract, profile, SystemSignals(
            queue_pressure=profile.queue_pressure, cache_pressure=profile.cache_pressure,
        ))), indent=2))
        return 0
    config = load_config(args.config)
    if args.command == "health":
        results = {name: provider.health().__dict__ if hasattr(provider, "health") else {"healthy": True, "detail": "not configured"} for name, provider in config.providers.items()}
        print(json.dumps(results, indent=2))
        return 0 if all(result["healthy"] for result in results.values()) else 1
    service = build_service(config)
    from .gateway import create_server
    print(f"LatencyOps gateway listening on http://{config.host}:{config.port}")
    create_server(config.host, config.port, service).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
