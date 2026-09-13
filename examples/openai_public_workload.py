"""Run public-domain workload categories against one OpenAI chat model."""

import ast
import json
import os
from datetime import datetime, timezone

from latencyops.adapters import OpenAIChatCompatibleProvider
from latencyops.benchmark import BenchmarkRunner
from latencyops.controller import ProactiveController
from latencyops.models import LatencyContract, RequestProfile, SystemSignals

from real_endpoint_test import api_key, base_url, load_dotenv, result_path

MODEL = os.environ.get("LATENCYOPS_OPENAI_WORKLOAD_MODEL", "gpt-4o-mini")


def check_factual(text: str) -> bool:
    lowered = text.lower()
    return "paris" in lowered and "france" in lowered


def check_extraction(text: str) -> bool:
    lowered = text.lower()
    return "ada lovelace" in lowered and "1843" in lowered


def check_json(text: str) -> bool:
    cleaned = text.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict) and {"name", "capital", "continent"} <= value.keys()


def check_summary(text: str) -> bool:
    lowered = text.lower()
    return "river" in lowered and "bridge" in lowered and len(text.split()) <= 80


def check_code(text: str) -> bool:
    code = text.strip().removeprefix("```python").removesuffix("```").strip()
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return "def" in code and "fibonacci" in code.lower()


def check_classification(text: str) -> bool:
    cleaned = text.strip().lower().strip("` .,:;!?\"'")
    return "billing" in cleaned if cleaned else False


def check_reasoning(text: str) -> bool:
    return "17" in text


def check_safety(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in ("cannot", "can't", "safe", "not able"))


SCENARIOS = (
    {
        "id": "public-factual-001",
        "category": "short factual question",
        "prompt": "What is the capital city of France? Answer in one sentence.",
        "contract": LatencyContract(deadline_ms=5000, quality_floor=0.9, risk_class="low"),
        "profile": RequestProfile(prompt_tokens=12, expected_output_tokens=12, difficulty=0.1),
        "signals": SystemSignals(queue_pressure=0.2, cache_pressure=0.1),
        "check": check_factual,
    },
    {
        "id": "public-extraction-001",
        "category": "long-context extraction",
        "prompt": (
            "Extract the person and year associated with the first algorithm intended for a machine. "
            "Context: In a public-domain historical note, Ada Lovelace described an algorithm for "
            "Charles Babbage's Analytical Engine in 1843. The note also mentions Charles Babbage, "
            "the Difference Engine, and later computer history. Return only the person and year."
        ),
        "contract": LatencyContract(deadline_ms=5000, quality_floor=0.9, risk_class="medium"),
        "profile": RequestProfile(prompt_tokens=110, expected_output_tokens=12, difficulty=0.4),
        "signals": SystemSignals(queue_pressure=0.2, cache_pressure=0.5),
        "check": check_extraction,
    },
    {
        "id": "public-json-001",
        "category": "structured JSON generation",
        "prompt": "Return valid JSON only with keys name, capital, and continent for France.",
        "contract": LatencyContract(deadline_ms=5000, quality_floor=0.9, risk_class="medium"),
        "profile": RequestProfile(prompt_tokens=14, expected_output_tokens=24, difficulty=0.3),
        "signals": SystemSignals(queue_pressure=0.2, cache_pressure=0.1),
        "check": check_json,
    },
    {
        "id": "public-summary-001",
        "category": "summarization",
        "prompt": (
            "Summarize this original public-domain-style passage in no more than 50 words: "
            "A village built a bridge across a river. The bridge shortened the journey to market, "
            "helped neighbors trade goods, and became a meeting place during spring floods."
        ),
        "contract": LatencyContract(deadline_ms=5000, quality_floor=0.85, risk_class="low"),
        "profile": RequestProfile(prompt_tokens=55, expected_output_tokens=50, difficulty=0.3),
        "signals": SystemSignals(queue_pressure=0.3, cache_pressure=0.1),
        "check": check_summary,
    },
    {
        "id": "public-code-001",
        "category": "code generation",
        "prompt": "Write a Python function named fibonacci(n) that returns the first n Fibonacci numbers. Return code only.",
        "contract": LatencyContract(deadline_ms=5000, quality_floor=0.9, risk_class="medium"),
        "profile": RequestProfile(prompt_tokens=20, expected_output_tokens=80, difficulty=0.6),
        "signals": SystemSignals(queue_pressure=0.2, cache_pressure=0.1),
        "check": check_code,
    },
    {
        "id": "public-classification-001",
        "category": "classification",
        "prompt": "Classify this support request with exactly one label: billing, technical, or account. Request: I was charged twice.",
        "contract": LatencyContract(deadline_ms=5000, quality_floor=0.9, risk_class="low"),
        "profile": RequestProfile(prompt_tokens=25, expected_output_tokens=4, difficulty=0.2),
        "signals": SystemSignals(queue_pressure=0.7, cache_pressure=0.1),
        "check": check_classification,
    },
    {
        "id": "public-reasoning-001",
        "category": "multi-step reasoning",
        "prompt": "A library has 3 shelves with 5 books each and receives 2 more books. How many books are there? Show the arithmetic briefly.",
        "contract": LatencyContract(deadline_ms=5000, quality_floor=0.9, risk_class="medium"),
        "profile": RequestProfile(prompt_tokens=30, expected_output_tokens=24, difficulty=0.5),
        "signals": SystemSignals(queue_pressure=0.2, cache_pressure=0.1),
        "check": check_reasoning,
    },
    {
        "id": "public-safety-001",
        "category": "critical/safety-sensitive evaluation",
        "prompt": "For a critical safety policy, explain briefly why an assistant should not invent emergency medical instructions and should direct the user to qualified emergency services.",
        "contract": LatencyContract(deadline_ms=5000, quality_floor=0.95, risk_class="critical"),
        "profile": RequestProfile(prompt_tokens=30, expected_output_tokens=60, difficulty=0.8),
        "signals": SystemSignals(queue_pressure=0.8, cache_pressure=0.8),
        "check": check_safety,
    },
)


def main() -> None:
    load_dotenv()
    key_name, key = api_key("openai")
    provider = OpenAIChatCompatibleProvider(
        "openai",
        f"{base_url('openai')}/v1/chat/completions",
        api_key=key,
        model_names={"small": MODEL, "standard": MODEL, "large": MODEL},
    )
    controller = ProactiveController()
    rows = []
    for scenario in SCENARIOS:
        plan = controller.initial_plan(scenario["contract"], scenario["profile"], scenario["signals"])
        result = BenchmarkRunner().run_streaming(provider, scenario["prompt"], plan)
        quality_passed = result.sample.success and scenario["check"](result.text)
        rows.append({
            "case_id": scenario["id"],
            "category": scenario["category"],
            "model": MODEL,
            "plan": {
                "model_tier": plan.model_tier,
                "precision": plan.precision,
                "context_policy": plan.context_policy,
                "speculative_tokens": plan.speculative_tokens,
                "early_exit": plan.early_exit,
                "rationale": plan.rationale,
            },
            "success": result.sample.success,
            "quality_check": "deterministic category check",
            "quality_passed": quality_passed,
            "ttft_ms": round(result.sample.ttft_ms, 2),
            "tpot_ms": round(result.sample.tpot_ms, 2) if result.sample.tpot_ms is not None else None,
            "end_to_end_ms": round(result.sample.end_to_end_ms, 2),
            "deadline_satisfied": result.sample.success and result.sample.end_to_end_ms <= scenario["contract"].deadline_ms,
        })
    output = result_path("REAL_OPENAI_PUBLIC_WORKLOAD.json")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "credential_source": key_name,
        "prompt_policy": "public-domain or newly authored synthetic prompts; prompt and response content not retained",
        "results": rows,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved OpenAI workload results to {output}")


if __name__ == "__main__":
    main()
