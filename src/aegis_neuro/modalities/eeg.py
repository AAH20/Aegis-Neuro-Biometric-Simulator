"""EEG modality (multi-channel spectral fingerprint).

The "Brain Print" template here is a per-channel band-power vector
computed in the canonical EEG bands (delta, theta, alpha, beta, gamma).
This is a reasonable abstraction of how published EEG biometric
systems (e.g., based on resting-state PSD or P300-evoked-potential
features) summarise a subject.

We do **not** simulate raw EEG time-series; we work in the feature
space the defender uses. This keeps the simulation deterministic,
fast, and explicitly scoped — we are modelling the *cryptographic
distance* of EEG biometrics, not their neuroscience.

Liveness model: defender checks that the candidate's inter-channel
power-ratio distribution is plausibly biological (channel ratios in
real recordings are highly correlated; random forgeries are not).
"""

from __future__ import annotations

import numpy as np

from .base import BiometricModality, Template

_CHANNELS = 8        # 10-20 system subset: Fp1 Fp2 C3 C4 O1 O2 T3 T4
_BANDS = 5           # delta, theta, alpha, beta, gamma


class EEGModality(BiometricModality):
    name = "eeg"
    default_threshold = 0.88

    def enroll(self, rng: np.random.Generator) -> Template:
        base = rng.lognormal(mean=0.0, sigma=0.4, size=(_CHANNELS, _BANDS))
        band_profile = np.array([3.0, 2.0, 4.0, 2.5, 1.5])
        base *= band_profile[None, :]
        subject_signature = rng.normal(0.0, 0.8, size=(_CHANNELS, _BANDS))
        features = base * np.exp(subject_signature)
        features /= np.linalg.norm(features) + 1e-12
        return Template(
            modality=self.name,
            features=features.astype(np.float64),
            metadata={"channels": _CHANNELS, "bands": _BANDS},
        )

    def score(self, candidate: Template, enrolled: Template) -> float:
        c = candidate.features.ravel()
        e = enrolled.features.ravel()
        if c.shape != e.shape:
            return 0.0
        c_centered = c - c.mean()
        e_centered = e - e.mean()
        denom = (np.linalg.norm(c_centered) * np.linalg.norm(e_centered)) + 1e-12
        cos = float(np.dot(c_centered, e_centered) / denom)
        return float(np.clip((cos + 1.0) / 2.0, 0.0, 1.0))

    def liveness(self, candidate: Template, rng: np.random.Generator) -> bool:
        f = candidate.features
        if f.shape != (_CHANNELS, _BANDS):
            return False
        channel_totals = f.sum(axis=1)
        if (channel_totals <= 0).any():
            return False
        ratios = f / channel_totals[:, None]
        std = float(ratios.std(axis=0).mean())
        return std < 0.18


__all__ = ["EEGModality"]
