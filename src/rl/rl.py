from abc import ABC, abstractmethod


class RLBase(ABC):

    def __init__(self, actions_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.98):
        self.action_space = actions_space  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.learn_count = 0
        self.choose_action_count = 0

    @abstractmethod
    def choose_action(self, observation, **kwargs):
        ...

    @abstractmethod
    def learn(self, status, action, reward, status_new, **kwargs):
        """
        learn function

        @param status:
        @param action:
        @param reward:
        @param status_new:
        @param kwargs:
          - terminated: x
          - truncated: xx
          - action_todo: xxx
        """
        ...

    @classmethod
    def _format_observation(cls, observation):
        return observation
