"""Rank continuation strategies for a model by actually running them.

`detect_strategy` answers "which strategy should this model use?" from the
registry and a set of name heuristics. This module answers the same question
empirically: run every candidate strategy over a small set of probe prefixes,
score each result with `scoring.score_continuation`, and rank them.

Costs a handful of short completions per strategy — see `basemode bench`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from .scoring import ContinuationScore, score_continuation

# Deliberately varied: a strategy that only holds up on plain narrative prose
# isn't the one you want pinned. Poetry and dialogue are where chat models
# most often break character and start explaining themselves.
DEFAULT_PROBES: tuple[str, ...] = (
    "The ship rounded the headland and",
    "To install the package, first ensure you have Python 3.11 or higher.",
    "the rain comes down like static\nbetween stations,",
    '"You knew," she said, not looking up. "You knew the whole time and you',
)

# The realistic chat-coercion space. `completion` and `fim` are excluded:
# they only apply to true base/code models, where there is nothing to rank.
DEFAULT_STRATEGIES: tuple[str, ...] = ("system", "prefill", "few_shot")

PROBE_TIMEOUT = 60  # seconds; a stalled provider stream shouldn't hang the run


@dataclass(frozen=True)
class ProbeResult:
    """One strategy run against one probe prefix."""

    prefix: str
    text: str
    score: ContinuationScore
    elapsed_s: float
    error: str | None = None


@dataclass(frozen=True)
class StrategyResult:
    """Every probe for one strategy, plus its aggregate score."""

    strategy: str
    probes: list[ProbeResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Mean continuation-purity score across probes (0.0 if all failed)."""
        if not self.probes:
            return 0.0
        return round(sum(p.score.score for p in self.probes) / len(self.probes), 3)

    @property
    def flags(self) -> tuple[str, ...]:
        """Every distinct flag raised across probes, most frequent first."""
        counts: dict[str, int] = {}
        for probe in self.probes:
            for flag in probe.score.flags:
                counts[flag] = counts.get(flag, 0) + 1
        return tuple(sorted(counts, key=lambda f: (-counts[f], f)))

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(p.error for p in self.probes if p.error)

    @property
    def mean_elapsed_s(self) -> float:
        if not self.probes:
            return 0.0
        return round(sum(p.elapsed_s for p in self.probes) / len(self.probes), 2)

    @property
    def sample(self) -> str:
        """First non-empty continuation, for eyeballing what the score means."""
        for probe in self.probes:
            if probe.text.strip():
                return probe.text.strip()
        return ""

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "score": self.score,
            "flags": list(self.flags),
            "errors": list(self.errors),
            "mean_elapsed_s": self.mean_elapsed_s,
            "sample": self.sample,
        }


async def _run_probe(
    model: str,
    strategy: str,
    prefix: str,
    *,
    max_tokens: int,
    temperature: float,
) -> ProbeResult:
    from .continue_ import continue_text
    from .observations import ObservationContext

    started = time.monotonic()

    async def collect() -> str:
        return "".join(
            [
                token
                async for token in continue_text(
                    prefix,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    strategy=strategy,
                    observation=ObservationContext(source="verification"),
                )
            ]
        )

    try:
        text = await asyncio.wait_for(collect(), timeout=PROBE_TIMEOUT)
    except TimeoutError:
        return ProbeResult(
            prefix=prefix,
            text="",
            score=score_continuation(prefix, ""),
            elapsed_s=round(time.monotonic() - started, 2),
            error=f"timed out after {PROBE_TIMEOUT}s",
        )
    except Exception as exc:  # provider rejected the strategy outright
        return ProbeResult(
            prefix=prefix,
            text="",
            score=score_continuation(prefix, ""),
            elapsed_s=round(time.monotonic() - started, 2),
            error=f"{type(exc).__name__}: {exc}",
        )

    return ProbeResult(
        prefix=prefix,
        text=text,
        score=score_continuation(prefix, text),
        elapsed_s=round(time.monotonic() - started, 2),
    )


async def bench_model(
    model: str,
    *,
    strategies: Sequence[str] = DEFAULT_STRATEGIES,
    probes: Sequence[str] = DEFAULT_PROBES,
    max_tokens: int = 60,
    temperature: float = 1.0,
) -> list[StrategyResult]:
    """Score every strategy for `model`, best first.

    Probes within a strategy run concurrently; strategies run one after
    another so a rate limit hits one row rather than poisoning the whole
    ranking. `temperature=1.0` is the one value every provider's sampler
    accepts, including models carrying the `no_temperature` quirk.
    """
    results: list[StrategyResult] = []
    for strategy in strategies:
        probe_results = await asyncio.gather(
            *(
                _run_probe(
                    model,
                    strategy,
                    prefix,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                for prefix in probes
            )
        )
        results.append(StrategyResult(strategy=strategy, probes=list(probe_results)))

    return rank(results)


def rank(results: list[StrategyResult]) -> list[StrategyResult]:
    """Sort by score, breaking ties toward the strategy that answered fastest."""
    return sorted(results, key=lambda r: (-r.score, r.mean_elapsed_s, r.strategy))


def winner(results: Sequence[StrategyResult]) -> StrategyResult | None:
    """Best strategy, or None when nothing produced a usable continuation."""
    ranked = rank(list(results))
    if not ranked or ranked[0].score <= 0:
        return None
    return ranked[0]
