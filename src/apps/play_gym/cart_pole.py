import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import FlattenObservation
from matplotlib import pyplot as plt
from tqdm import tqdm

from rl import utils
from rl.REINFORCE import REINFORCE
from rl.dqn import DQN
from rl.utils import ReplayBuffer


def dqnCartPole():
    env_name = 'CartPole-v0'
    num_episodes = 5000
    minimal_size = 500
    batch_size = 64

    env = FlattenObservation(gym.make(env_name, render_mode="human"))
    observation, info = env.reset()
    print(observation)
    input_dim = len(observation)

    agent = DQN(input_dim, env.action_space)
    replay_buffer = ReplayBuffer(capacity=batch_size * 100)
    return_list = []

    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                observation, info = env.reset()
                done = False
                while not done:
                    action = agent.choose_action(observation)
                    observation_new, reward, terminated, truncated, info, = env.step(action)
                    done = terminated or truncated

                    replay_buffer.add(observation, action, reward, observation_new, terminated, truncated)
                    observation = observation_new
                    episode_return += reward
                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        _state, _action, _reward, _state_new, _terminated, _truncated = replay_buffer.sample(batch_size)
                        agent.learn(_state, _action, _reward, _state_new, terminated=_terminated, truncated=_truncated)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    print('----------')
                    pbar.set_postfix({
                        'episode': (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

        episodes_list = list(range(len(return_list)))
        plt.plot(episodes_list, return_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DQN on {}'.format(env_name))
        plt.show()

        mv_return = utils.moving_average(return_list, 9)
        plt.plot(episodes_list, mv_return)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DQN on {}'.format(env_name))
        plt.show()


def learnReinforence():

    learning_rate = 1e-3
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98

    env_name = "CartPole-v1"
    env = FlattenObservation(gym.make(env_name, render_mode="human"))
    # env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]

    agent = REINFORCE(state_dim, env.action_space)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                states, actions, status_new, rewards = [], [], [], []
                while not done:
                    action = agent.choose_action(state)
                    observation_new, reward, terminated, truncated, info, = env.step(action)
                    states.append(state)
                    actions.append(action)
                    status_new.append(observation_new)
                    rewards.append(reward)
                    state = observation_new
                    episode_return += reward

                    if terminated or truncated:
                        done = True

                return_list.append(episode_return)
                agent.learn(states, actions, rewards, status_new)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

if __name__ == '__main__':
    # dqnCartPole()
    learnReinforence()
