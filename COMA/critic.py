#!/usr/bin/python3

import torch
import numpy as np
import torch.nn as nn

""" The Agent-dependent Critic. This is able to observe all states and actions and approximates the 
Q function for each action that an agent can take.
"""

class AdvantageCritic(torch.nn.Module):
    """
    Efficient implementation of global Q-function for action dependent baseline,
    input is joint state, fixed joint action excluding agent's action, one-hot agent index,
    outputs the Q-value for every action that agent can take (in one forward pass), while keeping other actions fixed
    """
    def __init__(self, input_size, hidden_size, action_size):

        super(AdvantageCritic, self).__init__()
        self.input_size = input_size
        self.output_size = action_size

        self.linear1 = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Tanh()
        )
        self.linear2 = nn.Linear(hidden_size, self.output_size)

    def forward(self, state_action):
        """
        :param state_action: joint action, global state, one-hot agent index
        :return: Q-value for the joint state
        """
        h = self.linear1(state_action)
        return self.linear2(h)

def unit_test():
    test_critic = AdvantageCritic(10, 256, 3)
    batch_size = 6
    state_action = torch.randn((batch_size, 10))
    output = test_critic.forward(state_action)
    if (output.shape == (batch_size, 3)):
        print("PASSED")

    print(output.shape)

if __name__ == "__main__":
    unit_test()

