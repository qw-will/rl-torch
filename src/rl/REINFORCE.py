import numpy as np
import torch
from gymnasium.wrappers import FlattenObservation
from torch import nn
from tqdm import tqdm

from rl.rl import RLBase
from rl.utils import device
import gymnasium as gym


class PolicyNet(nn.Module):

    def __init__(self, input_dim, action_dim, hidden_dim=None, **kwargs):
        super().__init__()
        self._input_dim = input_dim
        self._output_din = action_dim
        dim_h1 = input_dim * 10
        dim_h3 = action_dim * 10
        dim_h2 = int((dim_h1 + dim_h3) / 2)

        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(in_features=input_dim, out_features=dim_h1), nn.ReLU(),
            nn.Linear(in_features=dim_h1, out_features=dim_h2), nn.ReLU(),
            nn.Linear(in_features=dim_h2, out_features=dim_h3), nn.ReLU(),
            nn.Linear(in_features=dim_h3, out_features=self._output_din),
            nn.Softmax()
        )

    def forward(self, x):
        return self.net(x)


class REINFORCE(RLBase):

    def __init__(self, input_dim, actions_space, learning_rate=0.001, reward_decay=0.9, e_greedy=0.98):
        super().__init__(actions_space, learning_rate, reward_decay, e_greedy)
        self.input_dim = input_dim
        self.action_dim = action_dim = actions_space.n
        self.policy_net = PolicyNet(input_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)  # 使用Adam优化器
        self.device = device

    def choose_action(self, observation, **kwargs):
        state = torch.tensor([observation], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def learn(self, status, action, reward, status_new, **kwargs):
        reward_list = reward
        state_list = status
        action_list = action

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降

