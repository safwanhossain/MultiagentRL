#!/usr/bin/python3
import numpy as np
import multiagent.policy 
import torch
import torch.nn as nn

class Actor_Policy(Policy):
    ''' This is the class that the particle environment is used to. Requires the implementation
    of the action function which, given an observation, return an action'''
    def __init__(self, input_size, h_size, action_size):
        self.actor_network = Actor(input_size, h_size, action_size)

    def action(self, obs):
        actions = self.actor_network(obs, eps=0.1)
        
class Actor(torch.nn.Module):
    ''' This network, takes in observations, and returns an action. Action space is discreete, 
    5 possible actions (NOOP, LEFT, RIGHT, UP, DOWN)
    
    Observation space:
    Agent’s own velocity 2D
    Agent’s own position 2D
    Landmark positions with respect to the agent 3*2D
    The positions of other agents with respect to the agent 2*2D
    The messages C from other agents 2*2D messages (DISCARD)
    
    Note: Agents have access to almost everything about the global state except for other agent's 
    velocity. The GRU cell is still useful to model where other agents are going '''


    def __init__(self, input_size, h_size, action_size):
        super(Actor, self).__init__()
        self.input_size = input_size
        self.action_size = action_size
        self.h_size = h_size

        self.actor_gru = torch.nn.GRU(input_size=input_size,
                                      hidden_size=h_size,
                                      batch_first=True)

        self.linear = torch.nn.Linear(h_size, self.action_size)

    def forward(self, obs_seq, eps):
        """
        :param obs_seq: a sequence of shape (batch, seq_len, input_size)
        where input_size refers to size of [obs, prev_action]
        :param eps: softmax lower bound, for exploration
        :return:
        """
        batch_size = obs_seq.size()[0]
        h0 = torch.zeros(1, 10, self.h_size)

        # output has shape [batch, seq_len, h_size]
        output, hn = self.actor_gru(obs_seq, h0)
        logits = self.linear(output)

        # compute eps-bounded softmax
        softmax = torch.nn.functional.softmax(logits, dim=2)
        return (1 - eps) * softmax + eps / self.action_size

def unit_test():
    test_actor = Actor(input_size=14, h_size=128, action_size=5)
    obs_seq = torch.randn((10,6, 14))
    output = test_actor.forward(obs_seq, eps=0.01)
    if output.shape == (10, 6, 5):
        print("PASSED")

if __name__ == "__main__":
    unit_test()


