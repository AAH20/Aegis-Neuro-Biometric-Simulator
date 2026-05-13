"""Async multi-agent simulator.

The simulator wires :class:`AttackerAgent` -> :class:`DefenderAgent` ->
:class:`JudgeAgent` into a coroutine-driven loop:

    for attempt in 1..N:
        candidate = await attacker.propose()
        result    = await defender.authenticate(candidate)
        attacker.observe(candidate, result.similarity)
        judge.observe(record)
        if result.accepted: break

Even though the bodies are synchronous, expressing each step as a
coroutine keeps the architecture honest about the multi-agent
semantics and lets future strategies do real I/O (e.g., call out to a
local LLM scoring oracle) without rewriting the loop.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .agents import (
    AttackerAgent,
    AttackStrategy,
    AttemptRecord,
    DefenderAgent,
    JudgeAgent,
    JudgeVerdict,
)
from .modalities import BiometricModality, get_modality


@dataclass(slots=True)
class RunConfig:
    modality: str
    strategy: AttackStrategy
    max_attempts: int = 1_000
    seed: int = 1337
    leak_fraction: float = 0.20
    threshold: float | None = None
    check_liveness: bool = True
    population_size: int = 16
    stop_on_bypass: bool = True


@dataclass(slots=True)
class RunResult:
    config: RunConfig
    verdict: JudgeVerdict
    records: list[AttemptRecord]


async def run_simulation(config: RunConfig) -> RunResult:
    """Execute a single attacker-vs-defender simulation run."""
    rng = np.random.default_rng(config.seed)

    modality: BiometricModality = get_modality(config.modality)
    enrolled = modality.enroll(rng)

    attacker_rng = np.random.default_rng(config.seed + 1)
    defender_rng = np.random.default_rng(config.seed + 2)

    attacker = AttackerAgent(
        modality=modality,
        enrolled=enrolled,
        strategy=config.strategy,
        rng=attacker_rng,
        leak_fraction=config.leak_fraction,
        population_size=config.population_size,
    )
    defender = DefenderAgent(
        modality=modality,
        enrolled=enrolled,
        threshold=config.threshold,
        check_liveness=config.check_liveness,
        rng=defender_rng,
    )
    judge = JudgeAgent(
        modality=config.modality,
        strategy=config.strategy,
        max_attempts=config.max_attempts,
    )

    for attempt in range(1, config.max_attempts + 1):
        t0 = time.perf_counter()
        candidate = await attacker.propose()
        result = await defender.authenticate(candidate)
        attacker.observe(candidate, result.similarity)
        if config.strategy == "evolutionary":
            attacker.observe_evolutionary(candidate, result.similarity)
        elapsed = time.perf_counter() - t0

        record = AttemptRecord(
            attempt=attempt,
            modality=config.modality,
            strategy=config.strategy,
            similarity=result.similarity,
            threshold=result.threshold,
            liveness_passed=result.liveness_passed,
            accepted=result.accepted,
            wall_seconds=elapsed,
        )
        judge.observe(record)

        if result.accepted and config.stop_on_bypass:
            break

    return RunResult(
        config=config,
        verdict=judge.verdict(),
        records=judge.records,
    )


def run_simulation_sync(config: RunConfig) -> RunResult:
    """Synchronous helper for callers that don't want to manage a loop."""
    return asyncio.run(run_simulation(config))


async def run_many(configs: Iterable[RunConfig]) -> list[RunResult]:
    """Run multiple configurations concurrently in the same event loop."""
    return list(await asyncio.gather(*(run_simulation(c) for c in configs)))


__all__ = [
    "RunConfig",
    "RunResult",
    "run_many",
    "run_simulation",
    "run_simulation_sync",
]
