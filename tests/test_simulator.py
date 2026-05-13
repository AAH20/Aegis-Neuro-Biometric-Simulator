"""Integration tests for the full Attacker/Defender/Judge loop."""

from __future__ import annotations

from dataclasses import replace

import pytest

from aegis_neuro.simulator import RunConfig, run_simulation


pytestmark = pytest.mark.asyncio


async def test_blind_attack_rarely_bypasses_voice() -> None:
    cfg = RunConfig(
        modality="voice",
        strategy="blind",
        max_attempts=200,
        seed=1,
    )
    result = await run_simulation(cfg)
    assert not result.verdict.bypassed
    assert result.verdict.attempts == 200
    assert result.verdict.best_similarity < result.verdict.threshold


async def test_partial_leak_beats_blind_on_fingerprint() -> None:
    blind = await run_simulation(
        RunConfig(
            modality="fingerprint",
            strategy="blind",
            max_attempts=500,
            seed=10,
        )
    )
    leak = await run_simulation(
        RunConfig(
            modality="fingerprint",
            strategy="partial_leak",
            max_attempts=500,
            seed=10,
            leak_fraction=0.5,
        )
    )
    assert leak.verdict.best_similarity >= blind.verdict.best_similarity


async def test_hill_climb_improves_over_attempts_on_eeg() -> None:
    cfg = RunConfig(
        modality="eeg",
        strategy="hill_climb",
        max_attempts=200,
        seed=99,
        stop_on_bypass=False,
        check_liveness=False,
    )
    result = await run_simulation(cfg)
    early = max(r.similarity for r in result.records[:20])
    late = max(r.similarity for r in result.records[-20:])
    assert late >= early - 1e-6


async def test_judge_records_every_attempt() -> None:
    cfg = RunConfig(
        modality="ecg",
        strategy="evolutionary",
        max_attempts=64,
        seed=3,
        stop_on_bypass=False,
    )
    result = await run_simulation(cfg)
    assert len(result.records) == 64
    assert all(r.attempt == i + 1 for i, r in enumerate(result.records))


async def test_verdict_headline_is_human_readable() -> None:
    cfg = RunConfig(
        modality="retina",
        strategy="blind",
        max_attempts=100,
        seed=5,
    )
    result = await run_simulation(cfg)
    headline = result.verdict.headline
    assert isinstance(headline, str) and len(headline) > 5


async def test_seed_reproducibility() -> None:
    cfg = RunConfig(
        modality="voice",
        strategy="evolutionary",
        max_attempts=128,
        seed=2026,
        stop_on_bypass=False,
    )
    a = await run_simulation(cfg)
    b = await run_simulation(cfg)
    assert a.verdict.best_similarity == pytest.approx(b.verdict.best_similarity)
    assert a.verdict.attempts == b.verdict.attempts


async def test_liveness_wire_through_is_recorded() -> None:
    """Liveness state must be recorded on every AttemptRecord when
    check_liveness=True, regardless of whether it rejects."""
    cfg = RunConfig(
        modality="voice",
        strategy="blind",
        max_attempts=20,
        seed=12,
        check_liveness=True,
        stop_on_bypass=False,
    )
    result = await run_simulation(cfg)
    assert len(result.records) == 20
    assert all(isinstance(r.liveness_passed, bool) for r in result.records)
    assert all(r.liveness_passed for r in result.records), (
        "voice liveness should hold for blind unit-norm samples"
    )


async def test_liveness_disable_changes_outcome() -> None:
    """Disabling liveness must increase the *attack surface* — the
    bypass count must be at least as high without liveness as with."""
    base = RunConfig(
        modality="ecg",
        strategy="hill_climb",
        max_attempts=200,
        seed=21,
        stop_on_bypass=False,
    )
    with_liveness = await run_simulation(replace(base, check_liveness=True))
    without_liveness = await run_simulation(replace(base, check_liveness=False))
    accepted_with = sum(1 for r in with_liveness.records if r.accepted)
    accepted_without = sum(1 for r in without_liveness.records if r.accepted)
    assert accepted_without >= accepted_with
