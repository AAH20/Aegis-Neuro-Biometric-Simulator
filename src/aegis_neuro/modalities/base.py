"""Shared modality interface.

Every modality is responsible for:

1. Producing a *genuine* template from a seeded RNG (enrolment).
2. Producing a *similarity score* between a candidate template and the
   enrolment, in a way that respects a configured decision threshold.
3. Exposing the size of any "partial leak" an attacker is allowed to
   observe (e.g., a latent fingerprint residue exposes a few minutiae
   but not the full template).

Templates are passed around as opaque :class:`Template` objects so the
simulator never needs to know what shape a particular modality's
features actually are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class Template:
    """An opaque biometric template.

    The :attr:`features` field is modality-specific but always a numpy
    array so the Attacker can perturb it generically.
    """

    modality: str
    features: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "Template":
        return Template(
            modality=self.modality,
            features=self.features.copy(),
            metadata=dict(self.metadata),
        )


@dataclass(slots=True)
class MatchResult:
    """Result of comparing a candidate template against an enrolment."""

    similarity: float           # 0.0 .. 1.0 — higher = more similar
    accepted: bool              # similarity >= threshold AND liveness passed
    threshold: float
    liveness_passed: bool = True
    notes: str = ""


class BiometricModality:
    """Base class. Subclasses implement the modality-specific maths."""

    name: str = "abstract"
    default_threshold: float = 0.85

    def enroll(self, rng: np.random.Generator) -> Template:
        """Produce a genuine enrolment template."""
        raise NotImplementedError

    def score(self, candidate: Template, enrolled: Template) -> float:
        """Return a similarity score in [0.0, 1.0]."""
        raise NotImplementedError

    def liveness(self, candidate: Template, rng: np.random.Generator) -> bool:
        """Optional liveness / presentation-attack check.

        Default implementation accepts every candidate. Modalities override
        this to model defender liveness detectors.
        """
        return True

    def match(
        self,
        candidate: Template,
        enrolled: Template,
        *,
        threshold: float | None = None,
        rng: np.random.Generator | None = None,
        check_liveness: bool = False,
    ) -> MatchResult:
        """Full authentication decision."""
        thr = self.default_threshold if threshold is None else threshold
        similarity = self.score(candidate, enrolled)
        liveness_ok = (
            self.liveness(candidate, rng or np.random.default_rng())
            if check_liveness
            else True
        )
        accepted = bool(similarity >= thr and liveness_ok)
        return MatchResult(
            similarity=similarity,
            accepted=accepted,
            threshold=thr,
            liveness_passed=liveness_ok,
        )

    def leak_partial(
        self,
        template: Template,
        rng: np.random.Generator,
        leak_fraction: float,
    ) -> np.ndarray:
        """Return a partial leak of the template (defaults to a noisy sub-slice).

        ``leak_fraction`` is clamped to [0.0, 1.0]. Modalities may override
        this to model modality-specific side channels (latent residue,
        infrared reflections, recorded voice clip, etc.).
        """
        leak_fraction = float(np.clip(leak_fraction, 0.0, 1.0))
        n = template.features.size
        k = max(0, int(round(n * leak_fraction)))
        if k == 0:
            return np.zeros(0, dtype=template.features.dtype)
        flat = template.features.ravel()
        idx = rng.choice(n, size=k, replace=False)
        noise = rng.normal(0.0, 0.01, size=k).astype(flat.dtype)
        return flat[idx] + noise

    def random_candidate(self, rng: np.random.Generator) -> Template:
        """Produce a random candidate template (blind attack baseline)."""
        return self.enroll(rng)


__all__ = ["BiometricModality", "MatchResult", "Template"]
