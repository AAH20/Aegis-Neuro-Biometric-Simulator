"""Cross-run metric aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .agents.judge import JudgeVerdict
from .simulator import RunResult


@dataclass(slots=True)
class AggregateMetrics:
    """Summary across a fleet of runs (e.g., one per modality)."""

    runs: int
    bypassed: int
    held: int
    bypass_rate: float
    mean_attempts_to_bypass: float | None
    median_attempts_to_bypass: float | None
    mean_wall_ms_to_bypass: float | None
    weakest_modality: str | None
    strongest_modality: str | None


def aggregate(results: Iterable[RunResult]) -> AggregateMetrics:
    """Compute summary metrics across multiple :class:`RunResult` records."""
    results = list(results)
    if not results:
        return AggregateMetrics(
            runs=0,
            bypassed=0,
            held=0,
            bypass_rate=0.0,
            mean_attempts_to_bypass=None,
            median_attempts_to_bypass=None,
            mean_wall_ms_to_bypass=None,
            weakest_modality=None,
            strongest_modality=None,
        )

    bypassed_verdicts = [r.verdict for r in results if r.verdict.bypassed]
    held_verdicts = [r.verdict for r in results if not r.verdict.bypassed]

    mean_att = (
        sum(v.attempts_to_bypass or 0 for v in bypassed_verdicts)
        / len(bypassed_verdicts)
        if bypassed_verdicts
        else None
    )
    median_att = _median_or_none(
        [v.attempts_to_bypass for v in bypassed_verdicts if v.attempts_to_bypass]
    )
    mean_wall_ms = (
        sum((v.wall_seconds_to_bypass or 0.0) * 1000.0 for v in bypassed_verdicts)
        / len(bypassed_verdicts)
        if bypassed_verdicts
        else None
    )

    weakest = _weakest(bypassed_verdicts)
    strongest = _strongest(held_verdicts) or _strongest(bypassed_verdicts)

    return AggregateMetrics(
        runs=len(results),
        bypassed=len(bypassed_verdicts),
        held=len(held_verdicts),
        bypass_rate=len(bypassed_verdicts) / len(results),
        mean_attempts_to_bypass=mean_att,
        median_attempts_to_bypass=median_att,
        mean_wall_ms_to_bypass=mean_wall_ms,
        weakest_modality=weakest,
        strongest_modality=strongest,
    )


def _median_or_none(values: list[int]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2 == 1:
        return float(values[mid])
    return (values[mid - 1] + values[mid]) / 2.0


def _weakest(bypassed: list[JudgeVerdict]) -> str | None:
    if not bypassed:
        return None
    fastest = min(bypassed, key=lambda v: v.attempts_to_bypass or float("inf"))
    return fastest.modality


def _strongest(verdicts: list[JudgeVerdict]) -> str | None:
    if not verdicts:
        return None
    best = max(verdicts, key=lambda v: v.attempts)
    return best.modality


__all__ = ["AggregateMetrics", "aggregate"]
