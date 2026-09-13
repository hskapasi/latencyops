"""Public data contracts for the LatencyOps community edition."""

from dataclasses import dataclass, field
from typing import Literal

RiskClass = Literal["low", "medium", "high", "critical"]


@dataclass(frozen=True)
class LatencyContract:
    """Per-request latency, quality, and cost requirements."""

    deadline_ms: float
    ttft_target_ms: float | None = None
    tpot_target_ms: float | None = None
    quality_floor: float = 0.9
    risk_class: RiskClass = "medium"
    max_cost: float | None = None

    def __post_init__(self) -> None:
        if self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive")
        if not 0 <= self.quality_floor <= 1:
            raise ValueError("quality_floor must be between 0 and 1")
        if self.risk_class not in {"low", "medium", "high", "critical"}:
            raise ValueError("risk_class is invalid")


@dataclass(frozen=True)
class RequestProfile:
    """Workload features used by the planner without exposing prompt content."""

    prompt_tokens: int
    expected_output_tokens: int
    difficulty: float = 0.5
    draft_acceptance: float | None = None
    cache_pressure: float = 0.0
    queue_pressure: float = 0.0

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.expected_output_tokens < 0:
            raise ValueError("token counts cannot be negative")
        for name in ("difficulty", "cache_pressure", "queue_pressure"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.draft_acceptance is not None and not 0 <= self.draft_acceptance <= 1:
            raise ValueError("draft_acceptance must be between 0 and 1")


@dataclass(frozen=True)
class InferencePlan:
    """Explainable execution choices returned by the planner."""

    model_tier: Literal["small", "standard", "large"]
    precision: Literal["int4", "int8", "fp16"]
    context_policy: Literal["full", "selective", "compressed"]
    speculative_tokens: int
    early_exit: bool
    rationale: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ProviderCandidate:
    """Provider option with pre-request health and latency estimates."""

    name: str
    model_tier: str
    healthy: bool = True
    predicted_ttft_ms: float | None = None
    predicted_tpot_ms: float | None = None
    cost_per_1k_tokens: float | None = None


@dataclass(frozen=True)
class SystemSignals:
    """Content-free runtime signals available before planning."""

    queue_pressure: float = 0.0
    cache_pressure: float = 0.0
    provider_health: dict[str, bool] = field(default_factory=dict)
    predicted_ttft_ms: dict[str, float] = field(default_factory=dict)
    predicted_tpot_ms: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("queue_pressure", "cache_pressure"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if any(value < 0 for value in self.predicted_ttft_ms.values()):
            raise ValueError("predicted TTFT cannot be negative")
        if any(value < 0 for value in self.predicted_tpot_ms.values()):
            raise ValueError("predicted TPOT cannot be negative")


@dataclass(frozen=True)
class PolicyConfig:
    """Configurable thresholds for the rule-based proactive planner."""

    long_context_tokens: int = 8000
    high_queue_pressure: float = 0.75
    high_cache_pressure: float = 0.8
    speculation_acceptance: float = 0.75
    early_exit_difficulty: float = 0.35


@dataclass(frozen=True)
class QualityOutcome:
    """Content-free quality result supplied by an external evaluator."""

    score: float
    passed: bool

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 1:
            raise ValueError("quality score must be between 0 and 1")


@dataclass(frozen=True)
class StreamChunk:
    """Provider-neutral portion of a streamed response."""

    text: str
    token_count: int = 1

    def __post_init__(self) -> None:
        if self.token_count < 1:
            raise ValueError("token_count must be positive")


@dataclass(frozen=True)
class StreamingResult:
    """Collected streamed text and its measured latency sample."""

    text: str
    sample: "LatencySample"


@dataclass(frozen=True)
class LatencySample:
    """One request timing observation."""

    ttft_ms: float
    tpot_ms: float | None
    end_to_end_ms: float
    success: bool = True

    def __post_init__(self) -> None:
        if min(self.ttft_ms, self.end_to_end_ms) < 0:
            raise ValueError("latencies cannot be negative")
        if self.tpot_ms is not None and self.tpot_ms < 0:
            raise ValueError("tpot_ms cannot be negative")
