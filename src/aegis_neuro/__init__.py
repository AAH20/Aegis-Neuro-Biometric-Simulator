"""Aegis-Neuro.

A deterministic, local-only multi-agent adversarial simulator for
multi-modal biometric systems. Aegis-Neuro models a synthetic biometric
enrollment, an Attacker agent that crafts spoofed templates under one of
several strategies, a Defender agent that authenticates them, and a
Judge agent that records the attack/decision log and computes
time-to-bypass metrics.

The simulator is *intentionally simplified*. It does not break, and is
not designed to break, any real biometric device. It demonstrates
the **shape** of adversarial attacks against five common modalities
(fingerprint, retina, EEG, ECG, voice) so that defenders can reason
about FAR/FRR budgets, liveness detection, and multi-modal fusion at
the architectural level.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__: str = "0.1.0"
