from abc import ABC, abstractmethod

class Environment(ABC):

    class NoTerminationError(Exception):
        pass

    def __init__(self, end_condition, max_timesteps):
        self.reward = 0
        self.step_num = 0
        self.end_condition = end_condition
        self.max_timesteps = max_timesteps

    def reset(self):
        self.reward = 0
        self.step_num = 0

    @abstractmethod
    def step(self, actions):
        """
        Perform one timestep in the environment
        :param actions[List]: List of valid actions for each agent
        :return: List of observations for each agent (same indexing as actions)
        """
        self.step_num += 1
        if self.end_condition() or self.step_num > self.max_timesteps:
            raise self.NoTerminationError("The environment is running beyond termination conditions")


