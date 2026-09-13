"""Scenario benchmark runner for proactive policy evaluation."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import sleep
from typing import Callable

from .benchmark import BenchmarkRunner
from .models import InferencePlan, LatencyContract, LatencySample, RequestProfile
from .router import StreamingProvider


@dataclass(frozen=True)
class BenchmarkScenario:
    """A repeatable workload definition with content-free reporting metadata."""

    name: str
    prompt: str
    contract: LatencyContract
    profile: RequestProfile
    cache_state: str = "unknown"
    network_region: str = "unknown"
    estimated_cost_per_1k_tokens: float | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    """Execution controls for warm-up, concurrency, and optional rate limiting."""

    warmup_runs: int = 0
    measured_runs: int = 1
    concurrency: int = 1
    requests_per_second: float | None = None

    def __post_init__(self) -> None:
        if min(self.warmup_runs, self.measured_runs) < 0:
            raise ValueError("run counts cannot be negative")
        if self.concurrency < 1:
            raise ValueError("concurrency must be positive")
        if self.requests_per_second is not None and self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")


class ScenarioBenchmarkRunner:
    """Run identical scenarios concurrently and report contract outcomes."""

    def __init__(self, clock_runner: BenchmarkRunner | None = None):
        self.runner = clock_runner or BenchmarkRunner()

    def run(
        self,
        provider: StreamingProvider,
        scenario: BenchmarkScenario,
        plan: InferencePlan,
        config: BenchmarkConfig | None = None,
        quality_evaluator: Callable[[str], float] | None = None,
    ) -> dict[str, object]:
        config = config or BenchmarkConfig()
        for _ in range(config.warmup_runs):
            self.runner.run_streaming(provider, scenario.prompt, plan)
        interval = 1 / config.requests_per_second if config.requests_per_second else 0

        def measure(_: int) -> tuple[LatencySample, float | None]:
            if interval:
                sleep(interval)
            result = self.runner.run_streaming(provider, scenario.prompt, plan)
            quality = quality_evaluator(result.text) if quality_evaluator else None
            return result.sample, quality

        with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
            measured = list(executor.map(measure, range(config.measured_runs)))
        samples = [item[0] for item in measured]
        summary = self._summary(samples, scenario, measured)
        return summary

    @staticmethod
    def _summary(
        samples: list[LatencySample],
        scenario: BenchmarkScenario,
        measured: list[tuple[LatencySample, float | None]],
    ) -> dict[str, object]:
        successful = [sample for sample in samples if sample.success]
        quality_scores = [score for _, score in measured if score is not None]
        deadline_violations = sum(
            sample.success and sample.end_to_end_ms > scenario.contract.deadline_ms
            for sample in samples
        )
        tokens = scenario.profile.prompt_tokens + scenario.profile.expected_output_tokens
        cost = (
            tokens / 1000 * scenario.estimated_cost_per_1k_tokens
            if scenario.estimated_cost_per_1k_tokens is not None else None
        )
        from .metrics import percentile
        return {
            "scenario": scenario.name,
            "runs": len(samples),
            "warmup_excluded": True,
            "concurrency": len(samples),
            "success_rate": len(successful) / len(samples) if samples else 0.0,
            "ttft_p50_ms": percentile([s.ttft_ms for s in samples], 50) if samples else None,
            "ttft_p95_ms": percentile([s.ttft_ms for s in samples], 95) if samples else None,
            "tpot_p50_ms": percentile([s.tpot_ms for s in successful if s.tpot_ms is not None], 50) if any(s.tpot_ms is not None for s in successful) else None,
            "e2e_p50_ms": percentile([s.end_to_end_ms for s in samples], 50) if samples else None,
            "e2e_p95_ms": percentile([s.end_to_end_ms for s in samples], 95) if samples else None,
            "deadline_violations": deadline_violations,
            "quality_mean": sum(quality_scores) / len(quality_scores) if quality_scores else None,
            "quality_floor": scenario.contract.quality_floor,
            "quality_satisfied": all(score >= scenario.contract.quality_floor for score in quality_scores) if quality_scores else None,
            "estimated_cost": cost,
            "cache_state": scenario.cache_state,
            "network_region": scenario.network_region,
        }
