"""Tests for each biometric modality.

The contract every modality must satisfy:

* ``enroll(rng)`` is deterministic for a given RNG state.
* A self-match scores at least the modality's default threshold.
* Two different enrolments score *below* the default threshold most
  of the time (no trivial collisions).
* ``score`` is bounded to ``[0.0, 1.0]``.
* ``liveness`` returns True for the modality's own enrolments
  (otherwise no genuine user could ever be admitted).
"""

from __future__ import annotations

import numpy as np
import pytest

from aegis_neuro.modalities import MODALITY_REGISTRY, get_modality


MODALITY_NAMES = sorted(MODALITY_REGISTRY.keys())


@pytest.mark.parametrize("name", MODALITY_NAMES)
def test_enroll_is_deterministic(name: str) -> None:
    m = get_modality(name)
    t1 = m.enroll(np.random.default_rng(42))
    t2 = m.enroll(np.random.default_rng(42))
    assert np.allclose(t1.features, t2.features)
    assert t1.modality == name


@pytest.mark.parametrize("name", MODALITY_NAMES)
def test_self_match_is_perfect(name: str) -> None:
    m = get_modality(name)
    t = m.enroll(np.random.default_rng(7))
    score = m.score(t, t)
    assert score >= m.default_threshold
    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize("name", MODALITY_NAMES)
def test_different_enrolments_rarely_collide(name: str) -> None:
    m = get_modality(name)
    rng = np.random.default_rng(0)
    enrolled = m.enroll(rng)
    collisions = 0
    trials = 30
    for i in range(trials):
        other = m.enroll(np.random.default_rng(1000 + i))
        if m.score(other, enrolled) >= m.default_threshold:
            collisions += 1
    assert collisions <= 2, f"{name}: {collisions}/{trials} false matches"


@pytest.mark.parametrize("name", MODALITY_NAMES)
def test_score_bounded(name: str) -> None:
    m = get_modality(name)
    rng = np.random.default_rng(0)
    a = m.enroll(rng)
    b = m.enroll(np.random.default_rng(1))
    assert 0.0 <= m.score(a, b) <= 1.0
    assert 0.0 <= m.score(a, a) <= 1.0


@pytest.mark.parametrize("name", MODALITY_NAMES)
def test_liveness_passes_for_genuine_enrolments(name: str) -> None:
    m = get_modality(name)
    rng = np.random.default_rng(2024)
    passes = 0
    trials = 20
    for i in range(trials):
        t = m.enroll(np.random.default_rng(2024 + i))
        if m.liveness(t, rng):
            passes += 1
    assert passes >= trials - 2, f"{name}: only {passes}/{trials} live"


def test_unknown_modality_raises() -> None:
    with pytest.raises(ValueError):
        get_modality("does-not-exist")


def test_partial_leak_size_scales() -> None:
    m = get_modality("eeg")
    rng = np.random.default_rng(0)
    t = m.enroll(rng)
    leak_low = m.leak_partial(t, rng, 0.1)
    leak_high = m.leak_partial(t, rng, 0.8)
    assert leak_low.size <= leak_high.size
