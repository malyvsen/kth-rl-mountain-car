from dataclasses import dataclass
import numpy as np

from .basis_function import BasisFunction
from .hyperparameters import Hyperparameters


@dataclass(frozen=True)
class WeightedFunction:
    basis_function: BasisFunction
    hyperparameters: Hyperparameters
    weight: float
    eligibility: float

    def compute(self, observation: np.ndarray):
        return self.basis_function.compute(observation) * self.weight

    def train(self, error: float, observation: np.ndarray, action_selected: bool):
        eligibility = (
            self.eligibility
            * self.hyperparameters.discount
            * self.hyperparameters.forgetting
            + (self.basis_function.compute(observation) if action_selected else 0)
        )
        return type(self)(
            basis_function=self.basis_function,
            hyperparameters=self.hyperparameters,
            weight=self.weight
            + self.hyperparameters.learning_rate * error * eligibility,
            eligibility=eligibility,
        )
