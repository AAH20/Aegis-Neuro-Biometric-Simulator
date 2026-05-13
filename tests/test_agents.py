"""Unit tests for the individual agent classes."""

from __future__ import annotations

import numpy as np
import pytest

from aegis_neuro.agents import (
    ATTACK_STRATEGIES,
    AttackerAgent,
    AttemptRecord,
    DefenderAgent,
    JudgeAgent,
)
from aegis_neuro.modalities import get_modality


async def test_defender_accepts_self_match() -> None:
    m = get_modality("voice")
    enrolled = m.enroll(np.random.default_rng(0))
    defender = DefenderAgent(m, enrolled, check_liveness=False)
    result = await defender.authenticate(enrolled)
    assert result.accepted
    assert result.similarity >= result.threshold


async def test_defender_rejects_random() -> None:
    m = get_modality("voice")
    enrolled = m.enroll(np.random.default_rng(0))
    defender = DefenderAgent(m, enrolled, check_liveness=False)
    rejections = 0
    for i in range(20):
        other = m.enroll(np.random.default_rng(100 + i))
        result = await defender.authenticate(other)
        if not result.accepted:
            rejections += 1
    assert rejections >= 18


@pytest.mark.parametrize("strategy", ATTACK_STRATEGIES)
async def test_attacker_produces_template_of_right_modality(
    strategy: str,
) -> None:
    m = get_modality("eeg")
    enrolled = m.enroll(np.random.default_rng(0))
    attacker = AttackerAgent(
        modality=m,
        enrolled=enrolled,
        strategy=strategy,  # type: ignore[arg-type]
        rng=np.random.default_rng(1),
    )
    candidate = await attacker.propose()
    assert candidate.modality == "eeg"
    assert candidate.features.shape == enrolled.features.shape


def test_judge_records_and_summarises() -> None:
    judge = JudgeAgent(modality="ecg", strategy="blind", max_attempts=10)
    for i in range(5):
        judge.observe(
            AttemptRecord(
                attempt=i + 1,
                modality="ecg",
                strategy="blind",
                similarity=0.5 + i * 0.05,
                threshold=0.86,
                liveness_passed=True,
                accepted=False,
                wall_seconds=0.001,
            )
        )
    verdict = judge.verdict()
    assert verdict.attempts == 5
    assert verdict.bypassed is False
    assert verdict.best_similarity == pytest.approx(0.70)


def test_judge_records_first_bypass() -> None:
    judge = JudgeAgent(modality="voice", strategy="hill_climb", max_attempts=10)
    for i in range(7):
        accepted = (i == 4)
        judge.observe(
            AttemptRecord(
                attempt=i + 1,
                modality="voice",
                strategy="hill_climb",
                similarity=0.9 if accepted else 0.4,
                threshold=0.83,
                liveness_passed=True,
                accepted=accepted,
                wall_seconds=0.001,
            )
        )
    verdict = judge.verdict()
    assert verdict.bypassed is True
    assert verdict.attempts_to_bypass == 5
