"""Fingerprint modality (ANSI INCITS 378-style minutiae template).

A "template" is a set of N minutiae points; each minutia is encoded as
the triple ``(x, y, theta)`` in a normalised unit square with ``theta``
in radians. The matcher computes a soft assignment between the
candidate and the enrolment, weighted by spatial proximity and
orientation agreement.

Liveness model: detects implausibly tight clusters of points (a common
artefact of naive minutiae forgery on latent-residue lifts).
"""

from __future__ import annotations

import numpy as np

from .base import BiometricModality, Template

_N_MINUTIAE = 30
_FEATURES_PER_MINUTIA = 3  # x, y, theta


class FingerprintModality(BiometricModality):
    name = "fingerprint"
    default_threshold = 0.78

    def enroll(self, rng: np.random.Generator) -> Template:
        xs = rng.uniform(0.05, 0.95, size=_N_MINUTIAE)
        ys = rng.uniform(0.05, 0.95, size=_N_MINUTIAE)
        thetas = rng.uniform(0.0, 2 * np.pi, size=_N_MINUTIAE)
        features = np.stack([xs, ys, thetas], axis=1).astype(np.float64)
        return Template(modality=self.name, features=features)

    def score(self, candidate: Template, enrolled: Template) -> float:
        c = candidate.features
        e = enrolled.features
        if c.shape != e.shape:
            return 0.0

        cxy = c[:, :2]
        exy = e[:, :2]
        cth = c[:, 2]
        eth = e[:, 2]

        d2 = ((cxy[:, None, :] - exy[None, :, :]) ** 2).sum(axis=2)
        nn = d2.min(axis=1)
        nn_idx = d2.argmin(axis=1)

        spatial = np.exp(-nn / 0.004)

        dtheta = np.abs(cth - eth[nn_idx])
        dtheta = np.minimum(dtheta, 2 * np.pi - dtheta)
        orient = (np.cos(dtheta) + 1.0) / 2.0

        per_minutia = spatial * orient
        return float(np.clip(per_minutia.mean(), 0.0, 1.0))

    def liveness(self, candidate: Template, rng: np.random.Generator) -> bool:
        xy = candidate.features[:, :2]
        if xy.shape[0] < 4:
            return True
        d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)
        np.fill_diagonal(d2, np.inf)
        median_nn = float(np.median(np.sqrt(d2.min(axis=1))))
        return median_nn > 0.04

    def leak_partial(
        self,
        template: Template,
        rng: np.random.Generator,
        leak_fraction: float,
    ) -> np.ndarray:
        """A latent-residue leak: a fraction of the minutiae are exposed."""
        leak_fraction = float(np.clip(leak_fraction, 0.0, 1.0))
        n = template.features.shape[0]
        k = max(0, int(round(n * leak_fraction)))
        if k == 0:
            return np.zeros((0, _FEATURES_PER_MINUTIA), dtype=template.features.dtype)
        idx = rng.choice(n, size=k, replace=False)
        leaked = template.features[idx].copy()
        leaked[:, :2] += rng.normal(0.0, 0.01, size=(k, 2))
        leaked[:, 2] += rng.normal(0.0, 0.05, size=k)
        return leaked


__all__ = ["FingerprintModality"]
