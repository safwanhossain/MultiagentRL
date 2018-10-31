import numpy as np

class TestAgent:

    def __init__(self, num_enemies):
        self.step_num = 0
        self.num_enemies = num_enemies

    def reset(self):
        self.step_num = 0

    def step(self, obs):
        self.step_num += 1
        #self.step_num % 4
        return (1, [np.random.randint(64), np.random.randint(64), 0]) #np.random.randint(self.num_enemies)])