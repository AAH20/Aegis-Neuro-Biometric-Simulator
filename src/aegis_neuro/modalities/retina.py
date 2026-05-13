"""Retinal modality (vasculature-bifurcation template).

A retina template is encoded as N bifurcation points (junctions of
blood vessels) in a unit square. Each bifurcation has an associated
"branch angle" descriptor — the angle between its two child segments —
which a real retinal scanner uses to discriminate between similar
bifurcations.

Liveness model: detects implausibly uniform branch-angle distributions
(real retinas have a long-tailed angle distribution; naive forgeries
draw angles uniformly).
"""

from __future__ import annotations

import numpy as np

from .base import BiometricModality, Template

_N_BIFURCATIONS = 25
_FEATURES_PER_BIFURCATION = 3  # x, y, branch_angle


class RetinaModality(BiometricModality):
    name = "retina"
    default_threshold = 0.82

    def enroll(self, rng: np.random.Generator) -> Template:
        xs = rng.uniform(0.08, 0.92, size=_N_BIFURCATIONS)
        ys = rng.uniform(0.08, 0.92, size=_N_BIFURCATIONS)
        angles = rng.beta(2.0, 5.0, size=_N_BIFURCATIONS) * np.pi
        features = np.stack([xs, ys, angles], axis=1).astype(np.float64)
        return Template(modality=self.name, features=features)

    def score(self, candidate: Template, enrolled: Template) -> float:
        c = candidate.features
        e = enrolled.features
        if c.shape != e.shape:
            return 0.0
        cxy = c[:, :2]
        exy = e[:, :2]
        d2 = ((cxy[:, None, :] - exy[None, :, :]) ** 2).sum(axis=2)
        nn = d2.min(axis=1)
        nn_idx = d2.argmin(axis=1)
        spatial = np.exp(-nn / 0.005)

        dang = np.abs(c[:, 2] - e[nn_idx, 2])
        dang = np.minimum(dang, np.pi - dang)
        angular = np.exp(-dang / 0.25)

        per_bif = 0.7 * spatial + 0.3 * angular
        return float(np.clip(per_bif.mean(), 0.0, 1.0))

    def liveness(self, candidate: Template, rng: np.random.Generator) -> bool:
        angles = candidate.features[:, 2]
        if angles.size < 5:
            return True
        hist, _ = np.histogram(angles, bins=8, range=(0.0, np.pi))
        p = hist / max(1, hist.sum())
        p = p[p > 0]
        entropy = float(-(p * np.log2(p)).sum())
        return entropy < 2.9


__all__ = ["RetinaModality"]
