"""Compare configured OpenAI and Qwen endpoints without retaining content or secrets."""

import json
import os
from datetime import datetime, timezone

from latencyops.adapters import OpenAIChatCompatibleProvider
from latencyops.benchmark import BenchmarkRunner
from latencyops.metrics import summarize
from latencyops.models import InferencePlan

from real_endpoint_test import api_key, base_url, load_dotenv, result_path

DEFAULT_OPENAI_MODELS = ("gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini")
QWEN_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"


def configured_models() -> dict[str, list[str]]:
    openai_models = tuple(
        model.strip()
        for model in os.environ.get("LATENCYOPS_OPENAI_MODELS", ",".join(DEFAULT_OPENAI_MODELS)).split(",")
        if model.strip()
    )
    qwen_model = os.environ.get("LATENCYOPS_QWEN_MODEL", QWEN_MODEL)
    return {"openai": list(openai_models), "qwen": [qwen_model]}


def run_provider(name: str, model: str, runs: int = 3) -> dict[str, object]:
    key_name, key = api_key(name if name == "openai" else "litellm")
    provider = OpenAIChatCompatibleProvider(
        name,
        f"{base_url(name if name == 'openai' else 'litellm')}/v1/chat/completions",
        api_key=key,
        model_names={"small": model, "standard": model, "large": model},
        chat_template_kwargs={"enable_thinking": False} if name == "qwen" else None,
    )
    plan = InferencePlan("standard", "fp16", "full", 0, False)
    samples = [
        BenchmarkRunner().run_streaming(
            provider, "Say hello in one short sentence.", plan
        ).sample
        for _ in range(runs)
    ]
    return {
        "credential_source": key_name,
        "model": model,
        "runs": runs,
        "summary": summarize(samples),
    }


def main() -> None:
    load_dotenv()
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workload": "synthetic single-sentence streaming prompt; prompt content not retained",
        "providers": {
            provider: [run_provider(provider, model) for model in models]
            for provider, models in configured_models().items()
        },
    }
    output = result_path("REAL_ENDPOINT_COMPARISON.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved aggregate comparison to {output}")


if __name__ == "__main__":
    main()
