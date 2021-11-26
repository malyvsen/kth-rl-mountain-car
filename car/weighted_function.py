from dataclasses import dataclass, replace
import numpy as np

from .basis_function import BasisFunction
from .hyperparameters import Hyperparameters


@dataclass(frozen=True)
class WeightedFunction:
    basis_function: BasisFunction
    hyperparameters: Hyperparameters
    weight: float
    eligibility: float

    def reset(self) -> "WeightedFunction":
        return type(self)(
            basis_function=self.basis_function,
            hyperparameters=self.hyperparameters,
            weight=self.weight,
            eligibility=0,
        )

    def compute(self, observation: np.ndarray):
        return self.basis_function.compute(observation) * self.weight

    def train(self, error: float, observation: np.ndarray, action_selected: bool):
        eligibility = (
            self.eligibility
            * self.hyperparameters.discount
            * self.hyperparameters.forgetting
            + (self.weight_gradient(observation) if action_selected else 0)
        )
        return type(self)(
            basis_function=self.basis_function,
            hyperparameters=self.hyperparameters,
            weight=self.weight
            + self.hyperparameters.learning_rate
            * np.clip(
                error * eligibility,
                -self.hyperparameters.max_gradient,
                self.hyperparameters.max_gradient,
            ),
            eligibility=eligibility,
        )

    def weight_gradient(self, observation: np.ndarray) -> float:
        norm = np.sum(self.basis_function.coefficients ** 2) ** 0.5
        return self.basis_function.compute(observation) / max(norm, 1)

    def update_hyperparameters(
        self, hyperparameters: Hyperparameters
    ) -> "WeightedFunction":
        return replace(self, hyperparameters=hyperparameters)
