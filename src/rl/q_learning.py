from abc import ABC

import numpy as np
import pandas as pd

from rl.rl import RLBase


class QLearningTableBase(RLBase, ABC):
    def __init__(self, actions_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.98):

        super().__init__(actions_space, learning_rate, reward_decay, e_greedy)
        self.q_table = pd.DataFrame(columns=self.action_space, dtype=np.float64, )

    def choose_action(self, observation, **kwargs):
        observation = self._format_observation(observation)
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
            print(
                f'learn {self.learn_count} time,status:{observation},action:{action}, status action q:\n\t{state_action.to_string()}'
            )
        else:
            # choose random action
            action = np.random.choice(self.action_space)
            print(f'learn {self.learn_count} time, random action:{action}.')
        return action

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table.loc[state] = [0] * len(self.action_space)

    def show_q(self, episode=None, print_console=True):
        # jd = self.q_table.to_json(orient='records', lines=True)
        jd = self.q_table.to_string()
        if print_console:
            print(f'\nepisode: {episode},learn {self.learn_count} time, q table: \n{jd}\n')
        # sys.exit(0)
        return jd

    @classmethod
    def _format_observation(cls, observation):
        if not isinstance(observation, str):
            observation = str(observation)
        return observation


class QLearningTable(QLearningTableBase):
    def learn(self, status, action, reward, status_new, **kwargs):
        status = self._format_observation(status)
        status_new = self._format_observation(status_new)
        self.check_state_exist(status_new)
        q_predict = self.q_table.loc[status, action]
        if status_new != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[status_new, :].max()  # next state is not terminal
        else:
            q_target = reward  # next state is terminal
        self.q_table.loc[status, action] += self.lr * (q_target - q_predict)  # update

        self.learn_count += 1
        # self.show_q()


class SarsaTable(QLearningTableBase):

    def learn(self, status, action, reward, status_new, action_todo, **kwargs):
        status = self._format_observation(status)
        status_new = self._format_observation(status_new)
        self.check_state_exist(status)
        self.check_state_exist(status_new)

        q_predict = self.q_table.loc[status, action]
        if status_new != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[status_new, action_todo]  # next state is not terminal
        else:
            q_target = reward  # next state is terminal
        self.q_table.loc[status, action] += self.lr * (q_target - q_predict)  # update


class SarsaLambdaTable(QLearningTableBase):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.98, trace_decay=0.9):
        super().__init__(actions, learning_rate, reward_decay, e_greedy)

        # backward view, eligibility trace.
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        super().check_state_exist(state)
        # also update eligibility trace
        # self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, status, action, reward, status_new, action_todo, **kwargs):
        status = self._format_observation(status)
        status_new = self._format_observation(status_new)
        self.check_state_exist(status_new)
        q_predict = self.q_table.loc[status, action]
        if status_new != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[status_new, action_todo]  # next state is not terminal
        else:
            q_target = reward  # next state is terminal
        error = q_target - q_predict

        # increase trace amount for visited state-action pair

        # Method 1:
        # self.eligibility_trace.loc[s, a] += 1

        # Method 2:
        self.eligibility_trace.loc[status, :] *= 0
        self.eligibility_trace.loc[status, action] = 1

        # Q update
        self.q_table += self.lr * error * self.eligibility_trace

        # decay eligibility trace after update
        self.eligibility_trace *= self.gamma * self.lambda_
