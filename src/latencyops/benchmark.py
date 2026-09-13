"""Small benchmark runner for provider comparisons."""

from time import perf_counter
from typing import Callable, Iterable

from .models import InferencePlan, LatencySample, StreamingResult
from .router import InferenceProvider, StreamingProvider


class BenchmarkRunner:
    """Run synthetic or approved prompts without retaining prompt content."""

    def __init__(self, clock: Callable[[], float] = perf_counter):
        self.clock = clock

    def run(self, provider: InferenceProvider, prompts: Iterable[str], plan: InferencePlan) -> list[LatencySample]:
        samples: list[LatencySample] = []
        for prompt in prompts:
            started = self.clock()
            try:
                provider.generate(prompt, plan)
            except Exception:
                elapsed_ms = (self.clock() - started) * 1000
                samples.append(LatencySample(elapsed_ms, None, elapsed_ms, success=False))
                continue
            elapsed_ms = (self.clock() - started) * 1000
            samples.append(LatencySample(elapsed_ms, None, elapsed_ms))
        return samples

    def run_streaming(
        self, provider: StreamingProvider, prompt: str, plan: InferencePlan
    ) -> StreamingResult:
        """Collect one stream and measure TTFT and mean TPOT separately."""
        started = self.clock()
        first_chunk_at: float | None = None
        first_chunk_token_count = 0
        token_count = 0
        text: list[str] = []
        try:
            for chunk in provider.stream(prompt, plan):
                now = self.clock()
                if first_chunk_at is None:
                    first_chunk_at = now
                    first_chunk_token_count = chunk.token_count
                token_count += chunk.token_count
                text.append(chunk.text)
        except Exception:
            elapsed_ms = (self.clock() - started) * 1000
            return StreamingResult("".join(text), LatencySample(
                (first_chunk_at - started) * 1000 if first_chunk_at is not None else elapsed_ms,
                None,
                elapsed_ms,
                success=False,
            ))

        ended = self.clock()
        elapsed_ms = (ended - started) * 1000
        if first_chunk_at is None:
            return StreamingResult("", LatencySample(elapsed_ms, None, elapsed_ms))
        intervals = max(token_count - first_chunk_token_count, 0)
        tpot_ms = ((ended - first_chunk_at) * 1000 / intervals) if intervals else None
        return StreamingResult("".join(text), LatencySample(
            (first_chunk_at - started) * 1000, tpot_ms, elapsed_ms
        ))
