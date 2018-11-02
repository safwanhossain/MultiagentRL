#!/usr/bin/python3

import torch
import numpy as np
import torch.nn as nn

""" The Global, Centralized Critic. This is able to observe all states and actions and approximates the 
Q function for state, joint-action pair.

Inconsistencies are possible with the 'efficient' version of the critic, 
Use as target network (briefly mentioned in the appendix of the paper)
"""

class GlobalCritic(torch.nn.Module):
    """
    Vanilla global Q-function,
    input is joint state, joint action
    outputs the Q-value for that state-action pair
    """
    def __init__(self, input_size, hidden_size):

        super(GlobalCritic, self).__init__()
        self.input_size = input_size

        self.linear1 = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Tanh()
        )
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, state_action):
        """
        :param state_action: joint action, global state
        :return: Q-value for the joint state
        """
        h = torch.relu(self.linear1(state_action))
        h = torch.relu(self.linear2(h))
        return self.linear3(h)

def unit_test():
    test_critic = GlobalCritic(10, 256)
    batch_size = 6
    state_action = torch.randn((batch_size, 10))
    output = test_critic.forward(state_action)
    if (output.shape == (batch_size, 1)):
        print("PASSED")

    print(output.shape)

    print(output.grad_fn)
    output.backward(torch.ones((6, 1)))
    for param in test_critic.parameters():
        print(param.grad)

if __name__ == "__main__":
    unit_test()