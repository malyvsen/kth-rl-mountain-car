import gym
import itertools

from .agent import Agent


def train_episode(environment: gym.Env, agent: Agent, render=False):
    agent = agent.reset()
    state = environment.reset()
    action = agent.select_action(state)
    episode_reward = 0

    for step_id in itertools.count():
        if render:
            environment.render()
        next_state, reward, done, info = environment.step(action)
        next_action = agent.select_action(next_state)
        episode_reward += reward

        agent = agent.train(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            next_action=next_action,
            done=done,
        )

        if done:
            break

        state = next_state
        action = next_action

    if render:
        environment.close()
    return agent, episode_reward
