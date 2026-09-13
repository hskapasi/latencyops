"""Run proactive LatencyOps plans against real OpenAI-compatible endpoints."""

import json
from datetime import datetime, timezone

from latencyops.adapters import OpenAIChatCompatibleProvider
from latencyops.benchmark import BenchmarkRunner
from latencyops.controller import ProactiveController
from latencyops.models import LatencyContract, RequestProfile, SystemSignals

from real_endpoint_test import api_key, base_url, load_dotenv, result_path

MODELS = {
    "openai": "gpt-4o-mini",
    "qwen": "Qwen/Qwen3.6-35B-A3B-FP8",
}

SCENARIOS = (
    {
        "name": "easy-low-risk",
        "prompt": "Say hello in one short sentence.",
        "contract": LatencyContract(deadline_ms=5000, quality_floor=0.8, risk_class="low"),
        "profile": RequestProfile(prompt_tokens=8, expected_output_tokens=8, difficulty=0.1),
        "signals": SystemSignals(queue_pressure=0.8, cache_pressure=0.1),
    },
    {
        "name": "critical-high-difficulty",
        "prompt": "Provide a concise synthetic safety checklist.",
        "contract": LatencyContract(deadline_ms=5000, quality_floor=0.95, risk_class="critical"),
        "profile": RequestProfile(prompt_tokens=20, expected_output_tokens=24, difficulty=0.9),
        "signals": SystemSignals(queue_pressure=0.8, cache_pressure=0.9),
    },
    {
        "name": "long-context-cache-pressure",
        "prompt": "Summarize this synthetic context: " + "important context " * 80,
        "contract": LatencyContract(deadline_ms=5000, quality_floor=0.8, risk_class="medium"),
        "profile": RequestProfile(prompt_tokens=9000, expected_output_tokens=16, difficulty=0.5),
        "signals": SystemSignals(queue_pressure=0.2, cache_pressure=0.8),
    },
)


def run_endpoint(name: str, model: str) -> dict[str, object]:
    key_name, key = api_key("openai" if name == "openai" else "litellm")
    provider = OpenAIChatCompatibleProvider(
        name,
        f"{base_url(name if name == 'openai' else 'litellm')}/v1/chat/completions",
        api_key=key,
        model_names={"small": model, "standard": model, "large": model},
        chat_template_kwargs={"enable_thinking": False} if name == "qwen" else None,
    )
    controller = ProactiveController()
    rows = []
    for scenario in SCENARIOS:
        plan = controller.initial_plan(scenario["contract"], scenario["profile"], scenario["signals"])
        result = BenchmarkRunner().run_streaming(provider, scenario["prompt"], plan)
        rows.append({
            "scenario": scenario["name"],
            "plan": {
                "model_tier": plan.model_tier,
                "precision": plan.precision,
                "context_policy": plan.context_policy,
                "speculative_tokens": plan.speculative_tokens,
                "early_exit": plan.early_exit,
                "rationale": plan.rationale,
            },
            "success": result.sample.success,
            "ttft_ms": round(result.sample.ttft_ms, 2),
            "tpot_ms": round(result.sample.tpot_ms, 2) if result.sample.tpot_ms is not None else None,
            "end_to_end_ms": round(result.sample.end_to_end_ms, 2),
            "deadline_satisfied": result.sample.success and result.sample.end_to_end_ms <= scenario["contract"].deadline_ms,
            "quality_evaluation": "not configured",
        })
    return {"credential_source": key_name, "model": model, "scenarios": rows}


def main() -> None:
    load_dotenv()
    output = result_path("REAL_PROACTIVE_COMPARISON.json")
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workload": "synthetic proactive planning scenarios; prompt and response content not retained",
        "providers": {
            "openai": run_endpoint("openai", MODELS["openai"]),
            "qwen": run_endpoint("qwen", MODELS["qwen"]),
        },
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved proactive comparison to {output}")


if __name__ == "__main__":
    main()
