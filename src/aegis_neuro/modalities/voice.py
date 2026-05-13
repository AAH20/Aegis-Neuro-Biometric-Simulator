"""Voice modality (speaker-embedding fingerprint).

The template is a 128-dimensional unit-norm vector — a deliberate
abstraction of an x-vector / d-vector / ECAPA-TDNN speaker embedding.
The defender's score is cosine similarity, rescaled to [0, 1]. This
is the canonical comparison used by every production speaker
verification system.

Liveness model: defender checks that the candidate embedding sits on
the unit sphere (||v|| ≈ 1) — naive forgeries produced by random
sampling without normalisation fail this trivially. Real
presentation-attack detection is far more sophisticated; we model the
*shape* of the check, not the substance.
"""

from __future__ import annotations

import numpy as np

from .base import BiometricModality, Template

_DIM = 128


class VoiceModality(BiometricModality):
    name = "voice"
    default_threshold = 0.83

    def enroll(self, rng: np.random.Generator) -> Template:
        v = rng.normal(0.0, 1.0, size=_DIM)
        v /= np.linalg.norm(v) + 1e-12
        return Template(modality=self.name, features=v.astype(np.float64))

    def score(self, candidate: Template, enrolled: Template) -> float:
        c = candidate.features
        e = enrolled.features
        if c.shape != e.shape:
            return 0.0
        cn = np.linalg.norm(c) + 1e-12
        en = np.linalg.norm(e) + 1e-12
        cos = float(np.dot(c, e) / (cn * en))
        return float(np.clip((cos + 1.0) / 2.0, 0.0, 1.0))

    def liveness(self, candidate: Template, rng: np.random.Generator) -> bool:
        norm = float(np.linalg.norm(candidate.features))
        return 0.85 <= norm <= 1.15


__all__ = ["VoiceModality"]
