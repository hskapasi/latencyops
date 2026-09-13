"""Probe .env-configured OpenAI-compatible endpoints without printing secrets."""

import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from latencyops.adapters import OpenAIChatCompatibleProvider
from latencyops.models import LatencyContract, RequestProfile
from latencyops.policy import ElasticInferencePlanner
from latencyops.benchmark import BenchmarkRunner

DEFAULT_URLS = {
    "litellm": "https://llm-intdev.ai-ceo.ai",
    "openai": "https://api.openai.com",
}
KEY_NAMES = {
    "litellm": ("LATENCYOPS_LITELLM_API_KEY", "LITELLM_API_KEY", "LITELLM_MASTER_KEY", "QWEN_API_KEY"),
    "openai": ("LATENCYOPS_OPENAI_API_KEY", "OPENAI_API_KEY"),
}
URL_NAMES = {
    "litellm": ("LATENCYOPS_LITELLM_URL", "LITELLM_URL"),
    "openai": ("LATENCYOPS_OPENAI_URL", "OPENAI_BASE_URL"),
}


def env_file_path() -> Path:
    configured = os.environ.get("LATENCYOPS_ENV_FILE")
    return Path(configured).expanduser() if configured else Path.home() / ".latencyops" / ".env"


def result_path(filename: str) -> Path:
    configured = os.environ.get("LATENCYOPS_RESULTS_DIR")
    directory = Path(configured).expanduser() if configured else Path.home() / ".latencyops" / "results"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def load_dotenv() -> None:
    path = env_file_path()
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip("'\"")
        if name and name not in os.environ:
            os.environ[name] = value


def env_value(names: tuple[str, ...], default: str | None = None) -> str | None:
    return next((os.environ[name] for name in names if os.environ.get(name)), default)


def base_url(provider: str) -> str:
    value = env_value(URL_NAMES[provider], DEFAULT_URLS[provider]) or DEFAULT_URLS[provider]
    return value.rstrip("/").removesuffix("/v1")


def api_key(provider: str) -> tuple[str, str]:
    requested = os.environ.get("LATENCYOPS_KEY_NAME")
    names = KEY_NAMES[provider]
    if requested:
        if requested not in names or not os.environ.get(requested):
            raise RuntimeError(f"Requested credential variable {requested!r} is not set for {provider}")
        return requested, os.environ[requested]
    for name in names:
        if os.environ.get(name):
            return name, os.environ[name]
    raise RuntimeError(f"No {provider} API key variable was found in .env or the environment")


def get_models(url: str, key: str) -> list[str]:
    request = Request(
        f"{url}/v1/models",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        method="GET",
    )
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode())
    return [item["id"] for item in payload.get("data", []) if isinstance(item.get("id"), str)]


def smoke_test(provider_name: str, url: str, key: str, model: str) -> None:
    provider = OpenAIChatCompatibleProvider(
        provider_name,
        f"{url}/v1/chat/completions",
        api_key=key,
        model_names={"small": model, "standard": model, "large": model},
        chat_template_kwargs={"enable_thinking": False} if provider_name == "litellm" else None,
    )
    plan = ElasticInferencePlanner().plan(
        LatencyContract(deadline_ms=5000),
        RequestProfile(prompt_tokens=2, expected_output_tokens=8, difficulty=0.1),
    )
    result = BenchmarkRunner().run_streaming(provider, "Say hello in one short sentence.", plan)
    print(json.dumps({
        "model": model,
        "text": result.text,
        "ttft_ms": round(result.sample.ttft_ms, 2),
        "tpot_ms": round(result.sample.tpot_ms, 2) if result.sample.tpot_ms is not None else None,
        "end_to_end_ms": round(result.sample.end_to_end_ms, 2),
        "success": result.sample.success,
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("litellm", "openai"), default="litellm")
    parser.add_argument("--smoke", action="store_true", help="run one real chat completion")
    parser.add_argument("--model", help="model ID; defaults to LATENCYOPS_TEST_MODEL")
    args = parser.parse_args()
    load_dotenv()
    url = base_url(args.provider)
    key_name, key = api_key(args.provider)
    print(f"Endpoint: {url}")
    print(f"Credential source: {key_name}")
    try:
        models = get_models(url, key)
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise RuntimeError(
                f"The endpoint rejected {key_name}; rotate or verify the local credential"
            ) from exc
        raise RuntimeError(f"Model discovery failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError("Could not reach the configured LiteLLM endpoint") from exc
    print("Available models:")
    for model in models:
        print(model)
    if args.smoke:
        model = args.model or os.environ.get("LATENCYOPS_TEST_MODEL")
        if not model:
            raise RuntimeError("Set LATENCYOPS_TEST_MODEL or pass --model after reviewing the list")
        smoke_test(args.provider, url, key, model)


if __name__ == "__main__":
    main()
