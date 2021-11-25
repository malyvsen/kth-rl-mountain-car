from dataclasses import dataclass, replace
import gym
import itertools
import numpy as np
from typing import Dict, List

from .basis_function import BasisFunction
from .hyperparameters import Hyperparameters
from .weighted_function import WeightedFunction


@dataclass(frozen=True)
class Agent:
    environment: gym.Env
    hyperparameters: Hyperparameters
    random_action_probability: float
    weighted_functions: Dict[int, List[WeightedFunction]]

    @classmethod
    def random(
        cls,
        environment: gym.Env,
        hyperparameters: Hyperparameters,
        random_action_probability: float,
        function_order: int,
    ):
        return cls(
            environment=environment,
            hyperparameters=hyperparameters,
            random_action_probability=random_action_probability,
            weighted_functions={
                action: [
                    WeightedFunction.random(
                        basis_function=BasisFunction(
                            environment=environment, coefficients=np.array(coefficients)
                        ),
                        hyperparameters=hyperparameters,
                    )
                    for coefficients in itertools.product(
                        *[
                            range(function_order + 1)
                            for dimension in range(
                                environment.observation_space.shape[0]
                            )
                        ]
                    )
                ]
                for action in range(environment.action_space.n)
            },
        )

    def select_action(self, state: np.ndarray):
        if np.random.uniform() < self.random_action_probability:
            return np.random.choice(self.environment.action_space.n)
        return self.best_action(state)

    def train(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_action,
        done: bool,
    ) -> "Agent":
        target = reward + (
            0
            if done
            else self.hyperparameters.discount
            * self.action_value(next_state, next_action)
        )
        error = target - self.action_value(state, action)
        return replace(
            self,
            weighted_functions={
                updated_action: [
                    function.train(
                        error=error,
                        observation=state,
                        action_selected=action == updated_action,
                    )
                    for function in functions
                ]
                for updated_action, functions in self.weighted_functions.items()
            },
        )

    def best_action(self, state: np.ndarray) -> int:
        return max(
            range(self.environment.action_space.n),
            key=lambda action: self.action_value(state, action),
        )

    def action_value(self, state: np.ndarray, action: int) -> float:
        return sum(
            function.compute(state) for function in self.weighted_functions[action]
        )
