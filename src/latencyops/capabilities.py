"""Provider capability negotiation and safe plan enforcement."""

from dataclasses import dataclass

from .models import InferencePlan, LatencyContract


@dataclass(frozen=True)
class ProviderCapabilities:
    """Execution features a provider can honor."""

    streaming: bool = True
    model_mapping: bool = True
    precision: bool = False
    context_policy: bool = False
    speculation: bool = False
    early_exit: bool = False
    usage_metadata: bool = False
    queue_metrics: bool = False
    cache_metrics: bool = False


@dataclass(frozen=True)
class EnforcedPlan:
    """Requested plan, provider-supported plan, and unapplied choices."""

    requested: InferencePlan
    enforced: InferencePlan
    unsupported: tuple[str, ...] = ()


def enforce_plan(
    contract: LatencyContract,
    plan: InferencePlan,
    capabilities: ProviderCapabilities,
) -> EnforcedPlan:
    """Remove unsupported optimizations while preserving critical safeguards."""
    unsupported: list[str] = []
    precision = plan.precision
    context_policy = plan.context_policy
    speculative_tokens = plan.speculative_tokens
    early_exit = plan.early_exit
    rationale = list(plan.rationale)
    if precision != "fp16" and not capabilities.precision:
        unsupported.append("precision")
        precision = "fp16"
    if context_policy != "full" and not capabilities.context_policy:
        unsupported.append("context_policy")
        context_policy = "full"
    if speculative_tokens and not capabilities.speculation:
        unsupported.append("speculation")
        speculative_tokens = 0
    if early_exit and not capabilities.early_exit:
        unsupported.append("early_exit")
        early_exit = False
    if unsupported:
        rationale.append("provider cannot enforce: " + ", ".join(unsupported))
    enforced = InferencePlan(
        plan.model_tier, precision, context_policy, speculative_tokens, early_exit, tuple(rationale)
    )
    if contract.risk_class == "critical" and enforced != plan:
        enforced = InferencePlan(
            plan.model_tier, "fp16", "full", 0, False,
            tuple(rationale + ["critical safeguards enforced at provider boundary"]),
        )
    return EnforcedPlan(plan, enforced, tuple(unsupported))
