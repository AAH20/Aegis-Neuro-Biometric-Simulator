"""Biometric modality models used by Aegis-Neuro.

Each modality exposes the same interface (see :mod:`aegis_neuro.modalities.base`)
so the simulator can drive any of them interchangeably.

Modalities currently implemented:

* :mod:`aegis_neuro.modalities.fingerprint` — minutiae-point templates
* :mod:`aegis_neuro.modalities.retina` — vasculature-bifurcation graphs
* :mod:`aegis_neuro.modalities.eeg` — multi-channel spectral fingerprints
* :mod:`aegis_neuro.modalities.ecg` — P-QRS-T morphology + HRV
* :mod:`aegis_neuro.modalities.voice` — speaker-embedding vectors
"""

from __future__ import annotations

from .base import BiometricModality, MatchResult, Template
from .ecg import ECGModality
from .eeg import EEGModality
from .fingerprint import FingerprintModality
from .retina import RetinaModality
from .voice import VoiceModality

MODALITY_REGISTRY: dict[str, type[BiometricModality]] = {
    "fingerprint": FingerprintModality,
    "retina": RetinaModality,
    "eeg": EEGModality,
    "ecg": ECGModality,
    "voice": VoiceModality,
}


def get_modality(name: str) -> BiometricModality:
    """Instantiate a modality by short name."""
    try:
        return MODALITY_REGISTRY[name]()
    except KeyError as exc:
        raise ValueError(
            f"unknown modality {name!r}; known: {sorted(MODALITY_REGISTRY)}"
        ) from exc


__all__ = [
    "BiometricModality",
    "ECGModality",
    "EEGModality",
    "FingerprintModality",
    "MODALITY_REGISTRY",
    "MatchResult",
    "RetinaModality",
    "Template",
    "VoiceModality",
    "get_modality",
]
