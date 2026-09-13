"""Content-free experiments comparing proactive plans with a static baseline."""

from dataclasses import dataclass
from typing import Callable

from .benchmark import BenchmarkRunner
from .controller import ProactiveController
from .models import InferencePlan, LatencyContract, QualityOutcome, RequestProfile, SystemSignals
from .router import StreamingProvider


@dataclass(frozen=True)
class PolicyExperimentResult:
    """Aggregate comparison of two plans for one workload case."""

    selected: dict[str, object]
    baseline: dict[str, object]


class PolicyExperiment:
    """Run selected and baseline plans without retaining request content."""

    def __init__(self, controller: ProactiveController | None = None):
        self.controller = controller or ProactiveController()

    def run(
        self,
        provider: StreamingProvider,
        prompt: str,
        contract: LatencyContract,
        profile: RequestProfile,
        signals: SystemSignals | None = None,
        quality_evaluator: Callable[[str], QualityOutcome] | None = None,
    ) -> PolicyExperimentResult:
        shadow = self.controller.shadow(contract, profile, signals)
        runner = BenchmarkRunner()
        results = {}
        for name, plan in (("selected", shadow.selected), ("baseline", shadow.baseline)):
            result = runner.run_streaming(provider, prompt, plan)
            quality = quality_evaluator(result.text) if quality_evaluator else None
            results[name] = {
                "plan": plan,
                "sample": result.sample,
                "quality": quality,
                "deadline_satisfied": result.sample.success and result.sample.end_to_end_ms <= contract.deadline_ms,
                "quality_satisfied": quality is None or quality.score >= contract.quality_floor,
            }
        return PolicyExperimentResult(results["selected"], results["baseline"])
