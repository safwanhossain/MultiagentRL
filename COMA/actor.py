#!/usr/bin/python3
import numpy as np
import multiagent.policy 
import torch
import torch.nn as nn

class Actor_Policy():
    ''' This is the class that the particle environment is used to. Requires the implementation
    of the action function which, given an observation, return an action'''
    def __init__(self, input_size, h_size, action_size):
        self.actor_network = Actor(input_size, h_size, action_size)

    def action(self, obs):
        actions = self.actor_network(obs, eps=0.1)
        
class GRUActor(torch.nn.Module):
    ''' This network, takes in observations, and returns an action. Action space is discrete,
    '''

    def __init__(self, input_size, h_size, action_size):
        super(GRUActor, self).__init__()
        self.input_size = input_size
        self.action_size = action_size
        self.h_size = h_size

        self.actor_gru = torch.nn.GRU(input_size=input_size,
                                      hidden_size=h_size,
                                      batch_first=True)

        self.linear = torch.nn.Linear(h_size, self.action_size)

    def forward(self, obs_seq, eps):
        """
        outputs prob dist over possible actions, using an eps-bounded softmax for exploration
        input sequence shape is batch-first
        :param obs_seq: a sequence of shape (batch, seq_len, input_size)
        where input_size refers to size of [obs, prev_action]
        :param eps: softmax lower bound, for exploration
        :return:
        """
        batch_size = obs_seq.size()[0]
        # initial state, shape (num_layers * num_directions, batch, hidden_size)
        h0 = torch.zeros(1, batch_size, self.h_size)

        # output has shape [batch, seq_len, h_size]
        output, hn = self.actor_gru(obs_seq, h0)
        logits = self.linear(output)

        # compute eps-bounded softmax
        softmax = torch.nn.functional.softmax(logits, dim=2)
        return (1 - eps) * softmax + eps / self.action_size

class MLPActor(torch.nn.Module):

    def __init__(self, input_size, h_size, action_size):
        super(MLPActor, self).__init__()
        self.input_size = input_size
        self.action_size = action_size
        self.h_size = h_size

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, h_size),
            torch.nn.ReLU(),
            torch.nn.Linear(h_size, h_size),
            torch.nn.ReLU(),
            torch.nn.Linear(h_size, action_size))

    def forward(self, input, eps):
        """
        outputs prob dist over possible actions, using an eps-bounded softmax for exploration
        input sequence shape is batch-first
        :param obs_seq: a sequence of shape (batch, seq_len, input_size)
        where input_size refers to size of [obs, prev_action]
        :param eps: softmax lower bound, for exploration
        :return:
        """

        logits = self.model.forward(input)

        # compute eps-bounded softmax
        softmax = torch.nn.functional.softmax(logits, dim=2)
        return (1 - eps) * softmax + eps / self.action_size

def unit_test():
    test_actor = MLPActor(input_size=14, h_size=128, action_size=5)
    obs_seq = torch.randn((10, 6, 14))
    output = test_actor.forward(obs_seq, eps=0.01)
    if output.shape == (10, 6, 5):
        print("PASSED")

if __name__ == "__main__":
    unit_test()


