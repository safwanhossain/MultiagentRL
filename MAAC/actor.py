#!/usr/bin/python3
import numpy as np
import multiagent.policy 
import torch
import torch.nn as nn
import utils

class Actor_Policy():
    ''' This is the class that the particle environment is used to. Requires the implementation
    of the action function which, given an observation, return an action'''
    def __init__(self, input_size, action_size):
        self.actor_network = Actor(input_size, action_size)

    def get_network(self):
        return self.actor_network

    def action(self, obs):
        actions = self.actor_network(obs, eps=0.1)

class Actor_Network_Linear(torch.nn.Module):

    def __init__(self, input_size, action_size):
        super(Actor_Network_Linear, self).__init__()
        self.input_size = input_size
        self.action_size = action_size
        self.h_size = 256

        self.batch_norm = nn.BatchNorm1d(input_size, affine=False)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, self.h_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.h_size, self.h_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.h_size, action_size),
            torch.nn.Softmax(dim=1)
        )
        weight_init(mean=0.0, std=0.02)

    def weight_init(self, mean, std):
        for m in self._modules:
            utils.normal_init(self._modules[m], mean, std)
    
    def forward(self, observation, eps):
        """
        outputs prob dist over possible actions, using an eps-bounded softmax for exploration
        input sequence shape is batch-first
        :param obs_seq: a sequence of shape (batch, seq_len, input_size)
        where input_size refers to size of [obs, prev_action]
        :param eps: softmax lower bound, for exploration
        :return:
        """

        # Get a discrete probability  distribution over the action space
        softmax = self.model(self.batch_norm(observation))

        # compute eps-bounded softmax
        return (1 - eps) * softmax + eps / self.action_size

def unit_test():
    test_actor = Actor_Network_Linear(input_size=14, action_size=5)
    # Give here a batch of 10, each has a sequence of 6 actions
    obs_seq = torch.randn((10, 14))
    output = test_actor.forward(obs_seq, eps=0.01)
    if output.shape == (10, 5):
        print("PASSED")

if __name__ == "__main__":
    unit_test()
