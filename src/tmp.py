import copy
import json
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.wrappers import FlattenObservation

num_episodes = 10000

if __name__ == '__main__':
    env_name = 'FrozenLake-v1'

    env = FlattenObservation(gym.make(
        env_name, render_mode="human", desc=None, map_name="4x4", is_slippery=False, max_episode_steps=64
    ))

    obs = env.observation_space
    acs = env.action_space
    df = pd.DataFrame(columns=[_ for _ in range(4)], dtype=np.float32)
    for i in range(10000):
        df.loc[str(obs.sample()), acs.sample()] = i/100
    print(df)
    print(acs.sample())
