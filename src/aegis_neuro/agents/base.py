"""Shared agent base + attempt record schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AttemptRecord:
    """One attacker submission + defender decision."""

    attempt: int                  # 1-indexed within a single run
    modality: str
    strategy: str
    similarity: float
    threshold: float
    liveness_passed: bool
    accepted: bool
    wall_seconds: float
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "modality": self.modality,
            "strategy": self.strategy,
            "similarity": round(self.similarity, 6),
            "threshold": round(self.threshold, 6),
            "liveness_passed": self.liveness_passed,
            "accepted": self.accepted,
            "wall_seconds": round(self.wall_seconds, 6),
            "notes": self.notes,
        }


class Agent:
    """Async agent base class. Subclasses override :meth:`run`."""

    role: str = "agent"

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.role
        self.log: list[str] = []

    def _log(self, message: str) -> None:
        self.log.append(message)

    async def run(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass(slots=True)
class AgentState:
    """Optional shared agent state container used by the simulator."""

    extras: dict[str, Any] = field(default_factory=dict)


__all__ = ["Agent", "AgentState", "AttemptRecord"]
