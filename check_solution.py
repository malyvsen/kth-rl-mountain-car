# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 1 Problem 4 solution checker
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import pickle
from tqdm import trange

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
k = env.action_space.n      # tells you the number of actions
low, high = env.observation_space.low, env.observation_space.high


def scale_state_varibles(s, eta, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2
        and features transformation
    '''
    x = (s-low) / (high-low)
    return np.cos(np.pi * np.dot(eta, x))

def Qvalues(s, w):
    ''' Q Value computation '''
    return np.dot(w, s)

# Parameters
N_EPISODES = 50            # Number of episodes to run for trainings
CONFIDENCE_PASS = -135

# Fourier basis
p = 3
try:
    f = open('weights.pkl', 'rb')
    data = pickle.load(f)
    if 'W' not in data or 'N' not in data:
        print('Matrix W (or N) is missing in the dictionary.')
        exit(-1)
    w = data['W']
    eta = data['N']

    # Dimensionality checks
    if w.shape[1] != eta.shape[0]:
        print('m is not the same for the matrices W and N')
        exit(-1)
    m = w.shape[1]
    if w.shape[0] != k:
        print('The first dimension of W is not {}'.format(k))
        exit(-1)
    if eta.shape[1] != 2:
        print('The second dimension of eta is not {}'.format(2))
        exit(-1)
except:
    print('File weights.pkl not found!')
    exit(-1)



# Reward
episode_reward_list = []  # Used to store episodes reward


# Simulate episodes
print('Checking solution...')
EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
for i in EPISODES:
    EPISODES.set_description("Episode {}".format(i))
    # Reset enviroment data
    done = False
    state = scale_state_varibles(env.reset(), eta, low, high)
    total_episode_reward = 0.

    qvalues = Qvalues(state, w)
    action = np.argmax(qvalues)

    while not done:
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)
        next_state = scale_state_varibles(next_state, eta, low, high)
        qvalues_next = Qvalues(next_state, w)
        next_action = np.argmax(qvalues_next)

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        qvalues = qvalues_next
        action = next_action

    # Append episode reward
    episode_reward_list.append(total_episode_reward)

    # Close environment
    env.close()

avg_reward = np.mean(episode_reward_list)
confidence = np.std(episode_reward_list) * 1.96 / np.sqrt(N_EPISODES)


print('Policy achieves an average total reward of {:.1f} +/- {:.1f} with confidence 95%.'.format(
                avg_reward,
                confidence))

if avg_reward - confidence >= CONFIDENCE_PASS:
    print('Your policy passed the test!')
else:
    print('Your policy did not pass the test! The average reward of your policy needs to be greater than {} with 95% confidence'.format(CONFIDENCE_PASS))
