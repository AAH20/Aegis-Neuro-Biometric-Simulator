"""JudgeAgent — records every attempt and produces the run verdict."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .base import Agent, AttemptRecord


@dataclass(slots=True)
class JudgeVerdict:
    """Summary statistics for a complete simulation run."""

    modality: str
    strategy: str
    attempts: int
    max_attempts: int
    attempts_to_bypass: int | None
    wall_seconds_to_bypass: float | None
    total_wall_seconds: float
    best_similarity: float
    threshold: float
    bypassed: bool
    liveness_rejections: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def headline(self) -> str:
        if self.bypassed:
            assert self.attempts_to_bypass is not None
            assert self.wall_seconds_to_bypass is not None
            return (
                f"BYPASSED in {self.attempts_to_bypass} attempts / "
                f"{self.wall_seconds_to_bypass * 1000:.1f} ms wall"
            )
        return (
            f"HELD over {self.attempts} attempts "
            f"(best similarity {self.best_similarity:.3f} vs "
            f"threshold {self.threshold:.3f})"
        )


class JudgeAgent(Agent):
    role = "judge"

    def __init__(self, modality: str, strategy: str, max_attempts: int) -> None:
        super().__init__(name="judge")
        self.modality = modality
        self.strategy = strategy
        self.max_attempts = int(max_attempts)
        self.records: list[AttemptRecord] = []
        self._first_bypass: AttemptRecord | None = None

    def observe(self, record: AttemptRecord) -> None:
        self.records.append(record)
        if self._first_bypass is None and record.accepted:
            self._first_bypass = record

    def verdict(self) -> JudgeVerdict:
        if not self.records:
            return JudgeVerdict(
                modality=self.modality,
                strategy=self.strategy,
                attempts=0,
                max_attempts=self.max_attempts,
                attempts_to_bypass=None,
                wall_seconds_to_bypass=None,
                total_wall_seconds=0.0,
                best_similarity=0.0,
                threshold=0.0,
                bypassed=False,
                liveness_rejections=0,
            )

        best_sim = max(r.similarity for r in self.records)
        threshold = self.records[-1].threshold
        total_wall = sum(r.wall_seconds for r in self.records)
        liveness_rej = sum(
            1 for r in self.records if not r.liveness_passed
        )

        if self._first_bypass is not None:
            wall_to_bypass = sum(
                r.wall_seconds
                for r in self.records[: self._first_bypass.attempt]
            )
            return JudgeVerdict(
                modality=self.modality,
                strategy=self.strategy,
                attempts=len(self.records),
                max_attempts=self.max_attempts,
                attempts_to_bypass=self._first_bypass.attempt,
                wall_seconds_to_bypass=wall_to_bypass,
                total_wall_seconds=total_wall,
                best_similarity=best_sim,
                threshold=threshold,
                bypassed=True,
                liveness_rejections=liveness_rej,
            )
        return JudgeVerdict(
            modality=self.modality,
            strategy=self.strategy,
            attempts=len(self.records),
            max_attempts=self.max_attempts,
            attempts_to_bypass=None,
            wall_seconds_to_bypass=None,
            total_wall_seconds=total_wall,
            best_similarity=best_sim,
            threshold=threshold,
            bypassed=False,
            liveness_rejections=liveness_rej,
        )


__all__ = ["JudgeAgent", "JudgeVerdict"]
