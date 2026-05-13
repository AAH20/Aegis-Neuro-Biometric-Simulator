"""ECG modality (P-QRS-T morphology + heart-rate-variability fingerprint).

The template summarises an enrollee with a feature vector built from:

* P-wave amplitude and width (2 features)
* QRS amplitude, width, axis (3 features)
* T-wave amplitude, width (2 features)
* HRV summary: RMSSD, SDNN, mean RR (3 features)

— 10 features total, normalised. This intentionally mirrors the
feature menu used by several published ECG biometric papers without
committing to any one of them.

Liveness model: defender checks that the implied heart rate (from
mean RR) is within a physiological range (40–180 BPM).
"""

from __future__ import annotations

import numpy as np

from .base import BiometricModality, Template

_FEATURES = 10
_MEAN_RR_IDX = 9  # last feature is mean RR interval (ms)


class ECGModality(BiometricModality):
    name = "ecg"
    default_threshold = 0.86

    def enroll(self, rng: np.random.Generator) -> Template:
        p_amp = rng.normal(0.12, 0.02)
        p_width = rng.normal(0.10, 0.01)
        qrs_amp = rng.normal(1.20, 0.15)
        qrs_width = rng.normal(0.09, 0.01)
        qrs_axis = rng.normal(0.0, 0.3)
        t_amp = rng.normal(0.30, 0.05)
        t_width = rng.normal(0.16, 0.02)
        rmssd = rng.normal(40.0, 8.0)
        sdnn = rng.normal(55.0, 10.0)
        mean_rr = rng.normal(850.0, 60.0)
        feats = np.array(
            [
                p_amp, p_width, qrs_amp, qrs_width, qrs_axis,
                t_amp, t_width, rmssd, sdnn, mean_rr,
            ],
            dtype=np.float64,
        )
        return Template(modality=self.name, features=feats)

    def score(self, candidate: Template, enrolled: Template) -> float:
        if candidate.features.shape != enrolled.features.shape:
            return 0.0
        scales = np.array(
            [0.05, 0.02, 0.3, 0.02, 0.5, 0.1, 0.04, 15.0, 20.0, 100.0],
            dtype=np.float64,
        )
        diff = (candidate.features - enrolled.features) / scales
        dist = float(np.linalg.norm(diff) / np.sqrt(_FEATURES))
        return float(np.clip(np.exp(-dist), 0.0, 1.0))

    def liveness(self, candidate: Template, rng: np.random.Generator) -> bool:
        if candidate.features.size != _FEATURES:
            return False
        mean_rr = float(candidate.features[_MEAN_RR_IDX])
        if mean_rr <= 0:
            return False
        hr_bpm = 60_000.0 / mean_rr
        return 40.0 <= hr_bpm <= 180.0


__all__ = ["ECGModality"]
