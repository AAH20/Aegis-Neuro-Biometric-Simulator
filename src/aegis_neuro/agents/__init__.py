"""Aegis-Neuro agents.

The simulator orchestrates exactly three roles:

* :class:`AttackerAgent` — produces candidate biometric templates.
* :class:`DefenderAgent` — authenticates them.
* :class:`JudgeAgent` — observes both, records every attempt, and
  computes bypass metrics.

All three share the same :class:`Agent` async base class so the
simulator can drive them in a unified event loop.
"""

from __future__ import annotations

from .attacker import AttackerAgent, AttackStrategy, ATTACK_STRATEGIES
from .base import Agent, AttemptRecord
from .defender import DefenderAgent
from .judge import JudgeAgent, JudgeVerdict

__all__ = [
    "ATTACK_STRATEGIES",
    "Agent",
    "AttackStrategy",
    "AttackerAgent",
    "AttemptRecord",
    "DefenderAgent",
    "JudgeAgent",
    "JudgeVerdict",
]
