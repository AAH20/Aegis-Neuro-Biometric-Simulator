"""DefenderAgent — authenticates candidate templates."""

from __future__ import annotations

import numpy as np

from ..modalities.base import BiometricModality, MatchResult, Template
from .base import Agent


class DefenderAgent(Agent):
    role = "defender"

    def __init__(
        self,
        modality: BiometricModality,
        enrolled: Template,
        *,
        threshold: float | None = None,
        check_liveness: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(name="defender")
        self.modality = modality
        self.enrolled = enrolled
        self.threshold = modality.default_threshold if threshold is None else threshold
        self.check_liveness = check_liveness
        self.rng = rng or np.random.default_rng(0)

    async def authenticate(self, candidate: Template) -> MatchResult:
        return self.modality.match(
            candidate,
            self.enrolled,
            threshold=self.threshold,
            rng=self.rng,
            check_liveness=self.check_liveness,
        )


__all__ = ["DefenderAgent"]
