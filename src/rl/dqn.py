import collections
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from rl.rl import RLBase



class QNet(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(QNet, self).__init__()
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
        )

    def forward(self, x):
        return self.net(x)


class DQN(RLBase):

    def __init__(self, input_dim, actions_space, learning_rate=0.001, reward_decay=0.9, e_greedy=0.98):
        super(DQN, self).__init__(actions_space, learning_rate, reward_decay, e_greedy)
        self.input_dim = input_dim
        self.action_dim = action_dim = actions_space.n

        self.q_net = QNet(input_dim, action_dim)
        self.target_q_net = QNet(input_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.device = torch.device('cpu')
        self.target_update = 10

    def choose_action(self, observation, **kwargs):

        # action selection
        if np.random.uniform() < self.epsilon:
            state = torch.tensor([observation], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
            # print(
            #     f'q net: learn {self.learn_count} time, choose: {self.choose_action_count} times, status:{observation}, chose action:{action}'
            # )
        else:
            # choose random action
            # action = np.random.choice(self.action_space)
            action = np.random.randint(self.action_dim)
            # print(f'learn {self.learn_count} time, choose: {self.choose_action_count} times, random action:{action}.')
        self.choose_action_count += 1
        return action

    def learn(self, status, action, reward, status_new, **kwargs):
        states = torch.tensor(status, dtype=torch.float).to(self.device)
        actions = torch.tensor(action).view(-1, 1).to(self.device)
        rewards = torch.tensor(reward, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(status_new, dtype=torch.float).to(self.device)
        terminated, truncated = kwargs.get('terminated'), kwargs.get('truncated')
        is_over = [1 if terminated or truncated else 0 for terminated, truncated in zip(terminated, truncated)]
        dones = torch.tensor(is_over, dtype=torch.float).view(-1, 1).to(self.device)

        # Q值
        predict = self.q_net(states)
        q_values = predict.gather(1, actions)
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        self.learn_count += 1
        if self.learn_count % self.target_update == 0:
            # print(f'learn {self.learn_count} time, choose: {self.choose_action_count} times, update target q net')
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
