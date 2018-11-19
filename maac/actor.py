#!/usr/bin/python3
import numpy as np
import multiagent.policy 
import torch
import torch.nn as nn
from utils.initializations import normal_init, xavier_init


class Actor(torch.nn.Module):

    def __init__(self, obs_size, action_size, device):
        super(Actor, self).__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.h_size = 256
        self.device = device

        self.batch_norm = nn.BatchNorm1d(self.obs_size, affine=False)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.obs_size, self.h_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.h_size, self.h_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.h_size, action_size),
            torch.nn.Softmax(dim=1)
        )
        # self.weight_init(mean=0.0, std=0.02)
        self.model.apply(xavier_init)

        self.to(self.device)

    def forward(self, observation, eps, get_regularized=False):
        """
        outputs prob dist over possible actions, using an eps-bounded softmax for exploration
        input sequence shape is batch-first
        :param obs_seq: a sequence of shape (batch, seq_len, input_size)
        where input_size refers to size of [obs, prev_action]
        :param eps: softmax lower bound, for exploration
        :return:
        """

        # Get a discrete probability  distribution over the action space
        ret = observation if observation.shape[0] == 1 else self.batch_norm(observation)
        ret = self.model(ret)
        softmax_ret = nn.functional.softmax(ret)
        softmax_ret = (1 - eps) * softmax_ret + eps / self.action_size

        if get_regularized:
            return softmax_ret, (softmax_ret**2).mean(dim=1)
        else:
            return softmax_ret

    def get_params(self):
        return self.parameters()

def unit_test():
    test_actor = Actor(obs_size=14, action_size=5)
    # Give here a batch of 10, each has a sequence of 6 actions
    obs_seq = torch.randn((10, 14))
    output = test_actor.forward(obs_seq)
    if output.shape == (10, 5):
        print("PASSED")

if __name__ == "__main__":
    unit_test()

