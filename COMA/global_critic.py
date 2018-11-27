#!/usr/bin/python3

import torch
import numpy as np
import torch.nn as nn

""" The Global, Centralized Critic. This is able to observe all states and actions and approximates the 
Q function for state, joint-action pair.

Inconsistencies are possible with the 'efficient' version of the critic, 
"""

class GlobalCritic(torch.nn.Module):
    """
    Vanilla global Q-function,
    input is joint state, joint action
    outputs the Q-value for that state-action pair
    """
    def __init__(self, input_size, hidden_size, n_layers):
        """

        :param input_size:
        :param hidden_size:
        :param n_layers: number of hidden layers
        """

        super(GlobalCritic, self).__init__()
        self.input_size = input_size
        self.layers = [nn.Linear(self.input_size, hidden_size), nn.ReLU()]

        for n in range(n_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_size, 1))
        self.sequential = nn.Sequential(*self.layers)

        #self.sequential.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.normal(m.weight, mean=0, std=0.01)
            m.bias.data.fill_(0.001)

    def forward(self, state_action):
        """
        :param state_action: joint action, global state
        :return: Q-value for the joint state
        """
        return self.sequential(state_action)

def unit_test():
    test_critic = GlobalCritic(10, 256, 1)
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