import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation
from matplotlib import pyplot as plt
from tqdm import tqdm

from rl import utils
from rl.dqn import DQN
from rl.q_learning import QLearningTable, SarsaTable
from rl.utils import ReplayBuffer

# mmap = generate_random_map(size=19, p=0.8)
mmap = ["SHHHHG", "FFFFFF", "FFFFFF", "FFFFFF"]
env = gym.make('FrozenLake-v1', render_mode="human", desc=mmap, map_name="4x4", is_slippery=False,
               max_episode_steps=200)


def _update_FrozenLake_reward(observation, reward, terminated, truncated, observation_new, **kwargs):
    if terminated and reward == 0 or (observation_new == observation):
        reward += -100000
    elif reward == 1:
        reward += 100000
    else:
        reward += -1
    return reward


def qLearning():
    learner = QLearningTable(list(range(env.action_space.n)), e_greedy=0.9)
    max_episode = 100000000
    for episode in range(1, max_episode, 1):
        observation, info = env.reset()
        print(observation)

        while True:
            # action = env.action_space.sample()
            action = learner.choose_action(observation)
            # tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
            observation_new, reward, terminated, truncated, info, = env.step(action)
            # print(f"next state:{observation_new}, reward:{reward}, term:{terminated},trunc:{truncated}, info:{info}")

            if observation_new == observation:
                print('state not change')
            learner.learn(str(observation), action, reward, str(observation_new))
            observation = observation_new
            if terminated or truncated:
                break
        learner.show_q(episode)
    env.close()


def sarsaLearning():
    learner = SarsaTable(list(range(env.action_space.n)), e_greedy=0.9)
    max_episode = 100000000
    for episode in range(1, max_episode, 1):
        observation, info = env.reset()
        print(observation)

        action = learner.choose_action(observation)
        while True:
            # action = env.action_space.sample()
            # tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
            observation_new, reward, terminated, truncated, info, = env.step(action)
            # print(f"next state:{observation_new}, reward:{reward}, term:{terminated},trunc:{truncated}, info:{info}")
            if terminated and reward == 0 or (observation_new == observation):
                reward += -100000
            elif reward == 1:
                reward += 100000
            else:
                reward += -1

            if observation_new == observation:
                print('state not change')

            action_new = learner.choose_action(observation_new)
            learner.learn(observation, action, reward, observation_new, action_new)
            observation, action = observation_new, action_new
            if terminated or truncated:
                break
        learner.show_q(episode)
    env.close()


def dqnFrozenLakeLearning():
    num_episodes = 10000
    minimal_size = 1024
    batch_size = 256
    env_name = 'FrozenLake-v1'

    env = FlattenObservation(gym.make(
        env_name, render_mode="human", desc=None, map_name="4x4", is_slippery=False, max_episode_steps=64
    ))
    observation, info = env.reset()
    input_dim = len(observation)

    agent = DQN(input_dim, env.action_space)
    replay_buffer = ReplayBuffer(capacity=batch_size * 10)
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

                    if (terminated and reward == 0) or (observation_new == observation).all():
                        reward += -100
                    elif truncated:
                        reward += -50
                    elif reward == 1:
                        reward += 100
                    else:
                        reward += -1

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


if __name__ == '__main__':
    # qLearning()
    # sarsaLearning()
    dqnFrozenLakeLearning()