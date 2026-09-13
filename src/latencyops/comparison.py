"""Benchmark comparison and constraint-satisfaction helpers."""

from dataclasses import dataclass
from typing import Iterable

from .benchmark import BenchmarkRunner
from .metrics import summarize
from .models import InferencePlan, LatencyContract, LatencySample, RequestProfile
from .router import StreamingProvider


@dataclass(frozen=True)
class WorkloadCase:
    """Synthetic workload metadata; prompt content is never included in reports."""

    name: str
    prompt: str
    contract: LatencyContract
    profile: RequestProfile


def run_workload_matrix(
    provider: StreamingProvider,
    cases: Iterable[WorkloadCase],
    plan: InferencePlan,
    runner: BenchmarkRunner | None = None,
) -> list[dict[str, object]]:
    """Run cases and report measurements plus contract satisfaction."""
    runner = runner or BenchmarkRunner()
    rows = []
    for case in cases:
        result = runner.run_streaming(provider, case.prompt, plan)
        sample = result.sample
        rows.append({
            "case": case.name,
            "success": sample.success,
            "ttft_ms": sample.ttft_ms,
            "tpot_ms": sample.tpot_ms,
            "end_to_end_ms": sample.end_to_end_ms,
            "deadline_satisfied": sample.success and sample.end_to_end_ms <= case.contract.deadline_ms,
            "quality_floor": case.contract.quality_floor,
        })
    return rows


def compare(samples_by_provider: dict[str, list[LatencySample]]) -> list[dict[str, object]]:
    """Return stable, ranked summaries without retaining prompts or responses."""
    rows = []
    for provider, samples in samples_by_provider.items():
        summary = summarize(samples)
        rows.append({"provider": provider, **summary})
    return sorted(rows, key=lambda row: (row.get("e2e_p95_ms", float("inf")), row["provider"]))
