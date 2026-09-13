"""Dependency-free latency measurement and summary helpers."""

from dataclasses import asdict
from time import perf_counter
from typing import Callable, Iterable, TypeVar

from .models import LatencySample

T = TypeVar("T")


def percentile(values: Iterable[float], percentile_rank: float) -> float:
    """Return a linearly interpolated percentile for a non-empty collection."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("values must not be empty")
    if not 0 <= percentile_rank <= 100:
        raise ValueError("percentile_rank must be between 0 and 100")
    position = (len(ordered) - 1) * percentile_rank / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(samples: Iterable[LatencySample]) -> dict[str, float | int]:
    """Summarize request timing without retaining request content."""
    items = list(samples)
    if not items:
        raise ValueError("samples must not be empty")
    ttft = [item.ttft_ms for item in items]
    end_to_end = [item.end_to_end_ms for item in items]
    tpot = [item.tpot_ms for item in items if item.tpot_ms is not None]
    result: dict[str, float | int] = {
        "count": len(items),
        "success_rate": sum(item.success for item in items) / len(items),
        "ttft_p50_ms": percentile(ttft, 50),
        "ttft_p95_ms": percentile(ttft, 95),
        "e2e_p50_ms": percentile(end_to_end, 50),
        "e2e_p95_ms": percentile(end_to_end, 95),
    }
    if tpot:
        result["tpot_p50_ms"] = percentile(tpot, 50)
        result["tpot_p95_ms"] = percentile(tpot, 95)
    return result


def measure(call: Callable[[], T], ttft_ms: float | None = None) -> tuple[T, float]:
    """Measure total execution time around a provider call."""
    started = perf_counter()
    result = call()
    elapsed_ms = (perf_counter() - started) * 1000
    return result, ttft_ms if ttft_ms is not None else elapsed_ms
