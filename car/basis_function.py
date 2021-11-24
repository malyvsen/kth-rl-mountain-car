from dataclasses import dataclass
import gym
import numpy as np


@dataclass(frozen=True)
class BasisFunction:
    environment: gym.Env
    coefficients: np.ndarray

    def compute(self, observation: np.ndarray):
        activation = np.dot(self.coefficients, self.normalize(observation))
        return np.cos(np.pi * activation)

    def normalize(self, observation: np.ndarray):
        return (observation - self.environment.observation_space.low) / (
            self.environment.observation_space.high
            - self.environment.observation_space.low
        )
