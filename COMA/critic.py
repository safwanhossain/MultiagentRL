#!/usr/bin/python3

import torch
import numpy as np
import torch.nn as nn

""" The Global Critic. This is able to observe all states and actions and approximates the 
Q function.
"""

class Critic(torch.nn.Module):
    def __init__(self, input_size, hidden_size):

        super(Critic, self).__init__()
        self.input_size = input_size
        self.output_size = 1

        self.linear1 = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Tanh()
        )
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, state_action):
        """
        :param state_action: joint action, global state, one-hot agent index
        :return: Q-value for the joint state
        """
        h = torch.nn.functional.tanh(self.linear1(state_action))
        return self.linear2(h)

def unit_test():
    test_critic = Critic(10, 256)
    batch_size = 6
    state_action = torch.randn((batch_size, 10))
    output = test_critic.forward(state_action)
    if (output.shape == (batch_size, 1)):
        print("PASSED")

if __name__ == "__main__":
    unit_test()

