"""Explainable proactive latency-quality planning policies."""

from .models import (
    InferencePlan,
    LatencyContract,
    PolicyConfig,
    ProviderCandidate,
    RequestProfile,
    SystemSignals,
)


class AdaptiveProviderSelector:
    """Content-free adaptive selector using observed latency and quality outcomes."""

    def __init__(self):
        self._scores: dict[str, tuple[float, float]] = {}

    def observe(self, provider: str, sample_ms: float, quality_score: float | None = None) -> None:
        prior_latency, prior_quality = self._scores.get(provider, (sample_ms, quality_score or 1.0))
        self._scores[provider] = (
            prior_latency * 0.7 + sample_ms * 0.3,
            prior_quality * 0.7 + (quality_score if quality_score is not None else prior_quality) * 0.3,
        )

    def select(self, contract: LatencyContract, candidates: list[ProviderCandidate]) -> ProviderCandidate:
        eligible = [candidate for candidate in candidates if candidate.healthy]
        if not eligible:
            raise LookupError("no healthy inference providers available")
        def score(candidate: ProviderCandidate) -> tuple[float, float]:
            latency, quality = self._scores.get(candidate.name, (
                (candidate.predicted_ttft_ms or float("inf")) + (candidate.predicted_tpot_ms or float("inf")),
                1.0,
            ))
            quality_penalty = 0.0 if quality >= contract.quality_floor else float("inf")
            return quality_penalty + latency, candidate.cost_per_1k_tokens or float("inf")
        return min(eligible, key=score)


class PlanViolation(ValueError):
    """Raised when an execution plan violates a request contract."""


class ElasticInferencePlanner:
    """Select a conservative plan before execution from workload and runtime signals."""

    def __init__(self, config: PolicyConfig | None = None):
        self.config = config or PolicyConfig()

    def plan(
        self,
        contract: LatencyContract,
        profile: RequestProfile,
        signals: SystemSignals | None = None,
    ) -> InferencePlan:
        signals = signals or SystemSignals(
            queue_pressure=profile.queue_pressure,
            cache_pressure=profile.cache_pressure,
        )
        reasons: list[str] = []
        model_tier = "standard"
        precision = "fp16"
        context_policy = "full"
        speculative_tokens = 0
        early_exit = False

        if profile.difficulty >= 0.8 or contract.risk_class in {"high", "critical"}:
            model_tier = "large"
            reasons.append("preserve model capacity for difficult or high-risk requests")
        elif signals.queue_pressure >= self.config.high_queue_pressure or profile.difficulty <= 0.25:
            model_tier = "small"
            reasons.append("lower model tier for low difficulty or queue pressure")

        if profile.prompt_tokens >= self.config.long_context_tokens:
            if contract.risk_class in {"high", "critical"}:
                reasons.append("preserve full context for high-risk requests")
            else:
                context_policy = "compressed" if signals.cache_pressure >= 0.5 else "selective"
                reasons.append("reduce long-context memory and attention work")
        elif signals.cache_pressure >= self.config.high_cache_pressure and contract.risk_class != "critical":
            context_policy = "compressed"
            reasons.append("reduce cache pressure")

        if contract.risk_class == "critical":
            precision = "fp16"
        elif contract.risk_class == "high":
            precision = "int8" if signals.cache_pressure >= 0.7 else "fp16"
        elif signals.cache_pressure >= 0.4 or signals.queue_pressure >= 0.5:
            precision = "int8"
            reasons.append("reduce memory bandwidth under pressure")

        acceptance = profile.draft_acceptance
        if contract.risk_class != "critical" and profile.expected_output_tokens >= 32 and (acceptance is None or acceptance >= self.config.speculation_acceptance):
            speculative_tokens = 6 if acceptance is None else 8
            reasons.append("use speculative decoding for sufficiently long output")
        if profile.difficulty <= self.config.early_exit_difficulty and contract.risk_class == "low":
            early_exit = True
            reasons.append("allow early exit for easy low-risk requests")

        if contract.deadline_ms < 500 and model_tier == "large" and contract.risk_class == "low":
            model_tier = "standard"
            reasons.append("protect tight deadline with a standard model tier")

        plan = InferencePlan(model_tier, precision, context_policy, speculative_tokens, early_exit, tuple(reasons))
        self.validate(contract, plan)
        return plan

    def validate(self, contract: LatencyContract, plan: InferencePlan) -> None:
        """Reject plans that violate explicit quality and risk safeguards."""
        if contract.risk_class == "critical" and (plan.precision != "fp16" or plan.early_exit):
            raise PlanViolation("critical requests require FP16 and no early exit")
        if contract.quality_floor >= 0.95 and plan.early_exit:
            raise PlanViolation("high quality floors cannot use early exit")
        if contract.risk_class == "critical" and plan.context_policy != "full":
            raise PlanViolation("critical requests require full context")

    def select_provider(
        self,
        contract: LatencyContract,
        candidates: list[ProviderCandidate],
    ) -> ProviderCandidate:
        """Choose a healthy candidate predicted to meet the deadline."""
        eligible = [candidate for candidate in candidates if candidate.healthy]
        if not eligible:
            raise LookupError("no healthy inference providers available")
        predicted = [
            candidate for candidate in eligible
            if candidate.predicted_ttft_ms is not None
            and candidate.predicted_tpot_ms is not None
            and candidate.predicted_ttft_ms + candidate.predicted_tpot_ms <= contract.deadline_ms
        ]
        pool = predicted or eligible
        return min(pool, key=lambda candidate: (
            candidate.cost_per_1k_tokens if candidate.cost_per_1k_tokens is not None else float("inf"),
            (candidate.predicted_ttft_ms or float("inf")) + (candidate.predicted_tpot_ms or float("inf")),
        ))

    def safe_fallback(self, contract: LatencyContract, candidates: list[ProviderCandidate]) -> ProviderCandidate:
        """Return an approved healthy fallback; never silently downgrade critical work."""
        if contract.risk_class == "critical":
            candidates = [candidate for candidate in candidates if candidate.model_tier == "large"]
        return self.select_provider(contract, candidates)
