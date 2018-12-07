#!/usr/bin/python3
from utils.initializations import normal_init, xavier_init
import torch
import torch.nn as nn

class GRUActor(nn.Module):
    """
    This network, takes in observations, and returns an action. Action space is discrete
    """
    def __init__(self, input_size, h_size, action_size, device, n_agents):
        super(GRUActor, self).__init__()
        self.input_size = input_size
        self.action_size = action_size
        self.h_size = h_size
        self.device = device
        self.n_agents = n_agents
        self.h = [None for _ in range(self.n_agents)]

        self.actor_gru = nn.GRU(input_size=input_size,
                                      hidden_size=h_size,
                                      batch_first=True).to(self.device)

        self.linear = nn.Linear(h_size, self.action_size).to(self.device)
        self.apply(normal_init)
        self.to(self.device)

    def reset(self):
        self.h = [None for _ in range(self.n_agents)]

    def forward(self, obs_seq, n, eps):
        """
        outputs prob dist over possible actions, using an eps-bounded softmax for exploration
        input sequence shape is batch-first
        :param obs_seq: a sequence of shape (batch, seq_len, input_size)
        where input_size refers to size of [obs, prev_action]
        :param eps: softmax lower bound, for exploration
        :return:
        """
        # initial state, shape (num_layers * num_directions, batch, hidden_size)
        if self.h[n] is None:
            batch_size = obs_seq.size()[0]
            self.h[n] = torch.zeros(1, batch_size, self.h_size).to(self.device)

        # output has shape [batch, seq_len, h_size]
        output, self.h[n] = self.actor_gru(obs_seq, self.h[n])
        logits = self.linear(output)

        # compute eps-bounded softmax
        softmax = nn.functional.softmax(logits, dim=-1)
        return (1 - eps) * softmax + eps / self.action_size

class MLPActor(nn.Module):

    def __init__(self, input_size, h_size, action_size, device, n_agents):
        super(MLPActor, self).__init__()
        self.input_size = input_size
        self.action_size = action_size
        self.h_size = h_size
        self.device = device

        self.mlp = nn.Sequential(
            nn.Linear(input_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, action_size)
        )
        self.apply(normal_init)
        self.to(self.device)

    def reset(self):
        pass

    def forward(self, input, n, eps):
        """
        outputs prob dist over possible actions, using an eps-bounded softmax for exploration
        input sequence shape is batch-first
        :param obs_seq: a sequence of shape (batch, seq_len, input_size)
        where input_size refers to size of [obs, prev_action]
        :param eps: softmax lower bound, for exploration
        :return:
        """
        logits = self.mlp(input)

        # compute eps-bounded softmax
        softmax = nn.functional.softmax(logits, dim=-1)
        return (1 - eps) * softmax + eps / self.action_size

def unit_test():
    test_actor = GRUActor(input_size=14, h_size=128, action_size=5)
    obs_seq = torch.randn((10, 6, 14))
    output = test_actor.forward(obs_seq, eps=0.01)
    if output.shape == (10, 6, 5):
        print("PASSED")

if __name__ == "__main__":
    unit_test()


