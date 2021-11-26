import gym
from tqdm.auto import trange
from typing import List

from .agent import Agent
from .train import train_episode


def evaluate(
    environment: gym.Env, agent: Agent, num_episodes: int, loading_bar=True
) -> List[float]:
    return [
        train_episode(environment, agent)[1]
        for episode_id in (trange(num_episodes) if loading_bar else range(num_episodes))
    ]
