#!/usr/bin/python3
import numpy as np
import multiagent.policy 
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class Actor_Policy():
    ''' This is the class that the particle environment is used to. Requires the implementation
    of the action function which, given an observation, return an action'''
    def __init__(self, input_size, h_size, action_size):
        self.actor_network = Actor(input_size, h_size, action_size)

    def action(self, obs):
        actions = self.actor_network(obs, eps=0.1)

class Agent(ABC):

    def __init__(self, actor_net):
        self.actor_net = actor_net
        self.h_state = None

    @abstractmethod
    def get_action(self, actor_input, **args):
        pass

    @abstractmethod
    def get_action_dist(self, actor_input, **args):
        pass

    @abstractmethod
    def reset_state(self, actor_input, **args):
        pass

class GRUAgent(Agent):

    def __init__(self, actor_net):
        """
        :param actor_net: each agent gets an actor network,
        allows for easily switching between shared vs non-shared params
        """
        super(GRUAgent, self).__init__(actor_net)

        self.h_state = None

    def get_action(self, actor_input, **args):

        eps = args['eps']
        action_dist, state = self.actor_net.forward(actor_input, eps, self.h_state)
        self.h_state = state

        action_idx = (torch.multinomial(action_dist[0, 0, :], num_samples=1)).numpy()
        action = torch.zeros(self.actor_net.action_size)
        action[action_idx] = 1
        return action

    def get_action_dist(self, actor_input, **args):

        eps = args['eps']
        action_dist, state = self.actor_net.forward(actor_input, eps, self.h_state)
        self.h_state = state

        return action_dist

    def reset_state(self):
        self.h_state = None


class MLPAgent(Agent):
    def __init__(self, actor_net):
        """
        :param actor_net: each agent gets an actor network,
        allows for easily switching between shared vs non-shared params
        """
        super(MLPAgent, self).__init__(actor_net)

    def get_action(self, actor_input, **args):
        eps = args['eps']
        action_dist = self.actor_net.forward(actor_input, eps)

        action_idx = (torch.multinomial(action_dist[0, 0, :], num_samples=1)).numpy()
        action = torch.zeros(self.actor_net.action_size)
        action[action_idx] = 1
        return action

    def get_action_dist(self, actor_input, **args):
        eps = args['eps']
        action_dist = self.actor_net.forward(actor_input, eps)

        return action_dist

    def reset_state(self):
        pass

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

        self.actor_gru.apply(self.init_weights)
        #self.linear.apply(self.init_weights)

    def forward(self, obs_seq, eps, h_state=None):
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

        if h_state is None:
            h_state = torch.zeros(1, batch_size, self.h_size)

        # output has shape [batch, seq_len, h_size]
        output, h_state = self.actor_gru(obs_seq, h_state)
        logits = self.linear(output)

        # compute eps-bounded softmax
        softmax = torch.nn.functional.softmax(logits, dim=2)
        return (1 - eps) * softmax + eps / self.action_size, h_state

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.normal(m.weight, mean=0, std=0.01)
            m.bias.data.fill_(0.01)

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

        self.model.apply(self.init_weights)

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

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.normal(m.weight, mean=0, std=0.001)
            m.bias.data.fill_(0.01)

def unit_test():
    test_actor = GRUActor(input_size=14, h_size=128, action_size=5)
    obs_seq = torch.randn((10, 6, 14))
    output = test_actor.forward(obs_seq, eps=0.01, reset=True)
    if output.shape == (10, 6, 5):
        print("PASSED")

if __name__ == "__main__":
    unit_test()


