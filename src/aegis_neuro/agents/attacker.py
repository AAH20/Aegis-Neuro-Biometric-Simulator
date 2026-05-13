"""AttackerAgent — produces candidate templates against an enrolment.

Four strategies are implemented. Each strategy models a *class* of
attack capability, not a specific named exploit:

* ``blind``       — uniform random samples. Baseline.
* ``partial_leak``— attacker observes a fraction of the enrolment
                    (e.g., a latent fingerprint, a recorded voice
                    clip, an infrared retinal reflection) and
                    constructs candidates around the leaked subset.
* ``hill_climb``  — attacker observes the defender's similarity
                    score after each attempt and gradient-ascends.
                    Models a "verbose match-score oracle" exposure.
* ``evolutionary``— population-based optimiser over the feature
                    vector, using the score oracle. Models "AI-swarm"
                    optimisation, but deterministic given the seed.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Literal

import numpy as np

from ..modalities.base import BiometricModality, Template
from .base import Agent

AttackStrategy = Literal["blind", "partial_leak", "hill_climb", "evolutionary"]

ATTACK_STRATEGIES: tuple[AttackStrategy, ...] = (
    "blind",
    "partial_leak",
    "hill_climb",
    "evolutionary",
)


CandidateFactory = Callable[[], Template]
ScoreOracle = Callable[[Template], Awaitable[float]]


class AttackerAgent(Agent):
    role = "attacker"

    def __init__(
        self,
        modality: BiometricModality,
        enrolled: Template,
        *,
        strategy: AttackStrategy,
        rng: np.random.Generator,
        leak_fraction: float = 0.20,
        population_size: int = 16,
        score_oracle: ScoreOracle | None = None,
    ) -> None:
        super().__init__(name=f"attacker[{strategy}]")
        if strategy not in ATTACK_STRATEGIES:
            raise ValueError(
                f"unknown strategy {strategy!r}; known: {ATTACK_STRATEGIES}"
            )
        self.modality = modality
        self.enrolled = enrolled
        self.strategy: AttackStrategy = strategy
        self.rng = rng
        self.leak_fraction = float(np.clip(leak_fraction, 0.0, 1.0))
        self.population_size = int(max(2, population_size))
        self.score_oracle = score_oracle

        self._leaked: np.ndarray | None = None
        self._population: list[Template] = []
        self._population_scores: list[float] = []
        self._best_so_far: Template | None = None
        self._best_score: float = -1.0

    async def propose(self) -> Template:
        """Return the next candidate template to submit to the defender."""
        if self.strategy == "blind":
            return self._propose_blind()
        if self.strategy == "partial_leak":
            return self._propose_partial_leak()
        if self.strategy == "hill_climb":
            return await self._propose_hill_climb()
        if self.strategy == "evolutionary":
            return await self._propose_evolutionary()
        raise AssertionError(f"unhandled strategy {self.strategy}")

    def observe(self, candidate: Template, similarity: float) -> None:
        """Defender feedback hook (used by adaptive strategies)."""
        if similarity > self._best_score:
            self._best_score = similarity
            self._best_so_far = candidate.copy()

    def _propose_blind(self) -> Template:
        return self.modality.random_candidate(self.rng)

    def _propose_partial_leak(self) -> Template:
        if self._leaked is None:
            self._leaked = self.modality.leak_partial(
                self.enrolled, self.rng, self.leak_fraction
            )
        base = self.modality.random_candidate(self.rng)
        if self._leaked.size == 0:
            return base
        flat = base.features.ravel().copy()
        leak_flat = np.asarray(self._leaked).ravel()
        k = min(flat.size, leak_flat.size)
        flat[:k] = leak_flat[:k]
        base.features = flat.reshape(base.features.shape)
        return base

    async def _propose_hill_climb(self) -> Template:
        if self._best_so_far is None:
            seed = self.modality.random_candidate(self.rng)
            return seed
        candidate = self._best_so_far.copy()
        noise = self.rng.normal(0.0, 0.02, size=candidate.features.shape)
        candidate.features = candidate.features + noise
        return candidate

    async def _propose_evolutionary(self) -> Template:
        if not self._population:
            self._population = [
                self.modality.random_candidate(self.rng)
                for _ in range(self.population_size)
            ]
            self._population_scores = [-1.0] * self.population_size
            return self._population[0]

        if -1.0 in self._population_scores:
            idx = self._population_scores.index(-1.0)
            return self._population[idx]

        order = np.argsort(self._population_scores)[::-1]
        elites = [self._population[i].copy() for i in order[:2]]
        children: list[Template] = []
        for _ in range(self.population_size - 2):
            a, b = elites[0].features, elites[1].features
            mask = self.rng.random(size=a.shape) < 0.5
            child_feats = np.where(mask, a, b)
            child_feats = child_feats + self.rng.normal(0.0, 0.03, size=child_feats.shape)
            children.append(
                Template(modality=self.modality.name, features=child_feats)
            )

        self._population = elites + children
        self._population_scores = [-1.0] * self.population_size
        return self._population[0]

    def observe_evolutionary(self, candidate: Template, similarity: float) -> None:
        for i, member in enumerate(self._population):
            if member.features is candidate.features:
                self._population_scores[i] = similarity
                return


__all__ = ["ATTACK_STRATEGIES", "AttackStrategy", "AttackerAgent"]
