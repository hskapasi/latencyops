"""Closed-loop policy helpers for shadow evaluation and safe re-planning."""

from dataclasses import dataclass
from typing import Protocol

from .models import InferencePlan, LatencyContract, LatencySample, QualityOutcome, RequestProfile, SystemSignals
from .policy import ElasticInferencePlanner


class QualityEvaluator(Protocol):
    def evaluate(self, request_id: str, output: str) -> QualityOutcome:
        """Return a score without requiring the controller to retain the output."""
        ...


@dataclass(frozen=True)
class ShadowComparison:
    """Compare a selected plan with a baseline without storing request content."""

    selected: InferencePlan
    baseline: InferencePlan
    selected_sample: LatencySample | None = None
    baseline_sample: LatencySample | None = None
    selected_quality: QualityOutcome | None = None
    baseline_quality: QualityOutcome | None = None


@dataclass(frozen=True)
class ReplanDecision:
    """A safe plan transition based on observed aggregate outcomes."""

    plan: InferencePlan
    reason: str
    changed: bool


class ProactiveController:
    """Plan before execution, evaluate a baseline in shadow mode, and re-plan safely."""

    def __init__(self, planner: ElasticInferencePlanner | None = None):
        self.planner = planner or ElasticInferencePlanner()

    def initial_plan(
        self, contract: LatencyContract, profile: RequestProfile, signals: SystemSignals | None = None
    ) -> InferencePlan:
        return self.planner.plan(contract, profile, signals)

    def shadow(
        self,
        contract: LatencyContract,
        profile: RequestProfile,
        signals: SystemSignals | None = None,
    ) -> ShadowComparison:
        selected = self.planner.plan(contract, profile, signals)
        baseline = InferencePlan("standard", "fp16", "full", 0, False, ("static baseline",))
        self.planner.validate(contract, baseline)
        return ShadowComparison(selected=selected, baseline=baseline)

    def replan(
        self,
        contract: LatencyContract,
        profile: RequestProfile,
        current: InferencePlan,
        sample: LatencySample,
        quality: QualityOutcome | None = None,
        signals: SystemSignals | None = None,
    ) -> ReplanDecision:
        if quality is not None and quality.score < contract.quality_floor:
            safer = RequestProfile(
                profile.prompt_tokens, profile.expected_output_tokens, min(profile.difficulty + 0.3, 1.0),
                profile.draft_acceptance, profile.cache_pressure, profile.queue_pressure,
            )
            plan = self.planner.plan(contract, safer, signals)
            return ReplanDecision(plan, "quality outcome fell below the contract floor", plan != current)
        if contract.ttft_target_ms is not None and sample.ttft_ms > contract.ttft_target_ms:
            pressured = SystemSignals(
                queue_pressure=1.0,
                cache_pressure=(signals.cache_pressure if signals else profile.cache_pressure),
                provider_health=(signals.provider_health if signals else {}),
                predicted_ttft_ms=(signals.predicted_ttft_ms if signals else {}),
                predicted_tpot_ms=(signals.predicted_tpot_ms if signals else {}),
            )
            plan = self.planner.plan(contract, profile, pressured)
            return ReplanDecision(plan, "observed TTFT exceeded the request target", plan != current)
        return ReplanDecision(current, "current plan remains within observed constraints", False)
