#!/usr/bin/python3

import torch
import numpy as np
import torch.nn as nn
from utils.initializations import normal_init

""" The Agent-dependent Critic.Each critic (one per agent) is a global critic and can access
all observations from each agent (along with their actions). There is an attention layer for each
critic that will pay special attention to whatever is deemed important. In forward pass, the 
observations for all agents are passed in
"""

class Critic(nn.Module):
    """
    Although we have multiple global critics, this can be efficiently computed with a single module
    There are shared layers for all critics, and layers specific to each global critic. As such,
    it will return a Q value for each agent.
    """
    def __init__(self, observation_size, action_size, num_agents, attention_heads, embedding_dim, device):
        super(Critic, self).__init__()
        self.num_agents = num_agents
        self.action_size = action_size
        self.observation_size = observation_size
        self.attention_heads = attention_heads
        self.embedding_dim = embedding_dim
        self.attend_dim = self.embedding_dim // attention_heads
        self.device = device

        # unique to each agent's Q function
        # F functions will use the state only embedding and return the Q value for all possible states. The way they
        # implement this is fishy but for now (attention weights depend on action), we'll use theirs.
        self.g_functions = nn.ModuleList()
        self.state_only_embeddings = nn.ModuleList()
        self.f_functions = nn.ModuleList()
        for i in range(self.num_agents):
            self.g_functions.append(nn.Sequential(
                # nn.BatchNorm1d(observation_size+action_size),
                nn.Linear(observation_size+action_size, self.embedding_dim),
                nn.LeakyReLU()
            ))
            self.state_only_embeddings.append(nn.Sequential(
                # nn.BatchNorm1d(observation_size),
                nn.Linear(observation_size, self.embedding_dim),
                nn.LeakyReLU()
            ))
            self.f_functions.append(nn.Sequential(
                nn.Linear(self.embedding_dim + 
                    self.attend_dim*self.attention_heads, self.embedding_dim),
                nn.LeakyReLU(),
                nn.Linear(self.embedding_dim, self.action_size)
            ))

        # Shared across all critics 
        # Matrix mult with a vector can be represented as a single linear layer
        self.Wq_layers = nn.ModuleList()
        self.Wk_layers = nn.ModuleList()
        self.V_layers = nn.ModuleList()
        for i in range(self.attention_heads):
            self.Wq_layers.append(nn.Linear(self.embedding_dim, self.attend_dim, bias=False))
            self.Wk_layers.append(nn.Linear(self.embedding_dim, self.attend_dim, bias=False))
            self.V_layers.append(nn.Linear(self.embedding_dim, self.attend_dim, bias=False))
        self.weight_init(mean=0.0, std=0.02)

        self.to(self.device)
    
    def get_non_attention_parameters(self):
        return (p for n, p in self.named_parameters() if 'layers' in n)
    
    def get_attention_parameters(self):
        return (p for n, p in self.named_parameters() if 'functions' in n or 'embeddings' in n)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m])

    def forward(self, observation_action_vector, regularize=False, ret_all_actions=False):
        """
        observation vector is the observation for each agent. We are NOT using some global
        state representations since it can't really work well with this attention model
        """
        observation_vector = observation_action_vector[0]
        action_vector = observation_action_vector[1]
        out_shape = observation_vector.shape[1:3]   # batch size
        assert (observation_vector.shape == (self.num_agents, *out_shape, self.observation_size))
        assert (action_vector.shape == (self.num_agents, *out_shape, self.action_size))

        # First compute the embedding
        ei_s = []
        si_s = []
        for i in range(self.num_agents):
            combined = torch.cat([observation_vector[i], action_vector[i]], dim=2)
            ei_s.append(self.g_functions[i](combined))
            si_s.append(self.state_only_embeddings[i](observation_vector[i]))

        all_queries = [[self.Wq_layers[l](ei_s[i]) for i in range(self.num_agents)] for l in
                       range(self.attention_heads)]
        all_keys = [[self.Wk_layers[l](ei_s[i]) for i in range(self.num_agents)] for l in range(self.attention_heads)]
        all_values = [[torch.nn.functional.leaky_relu(self.V_layers[l](ei_s[i])) \
                       for i in range(self.num_agents)] for l in range(self.attention_heads)]

        xi_s = [[] for _ in range(self.num_agents)]
        all_attend_logits = [[] for _ in range(self.num_agents)]
        for i in range(self.attention_heads):
            if self.num_agents == 1:
                xi_s[0].append(torch.zeros(*out_shape, self.attend_dim).to(self.device))
                continue
            head_queries = all_queries[i]
            head_keys = all_keys[i]
            head_values = all_values[i]

            for j in range(self.num_agents):
                query = head_queries[j]
                keys = [k for l, k in enumerate(head_keys) if l != j]
                values = [v for l, v in enumerate(head_values) if l != j]

                # calculate attention across agents
                attend_logits = torch.matmul(query.view(*out_shape, 1, self.attend_dim),
                                             torch.stack(keys).permute(1, 2, 3, 0))
                all_attend_logits[j].append(attend_logits)

                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                alpha = torch.nn.functional.softmax(scaled_attend_logits, dim=3)
                x_i = (torch.stack(values).permute(1, 2, 3, 0) * alpha).sum(dim=3)
                xi_s[j].append(x_i)

        all_action_q_for_agent = torch.zeros(self.num_agents, *out_shape, self.action_size).to(self.device)
        curr_action_q_for_agent = torch.zeros(self.num_agents, *out_shape, 1).to(self.device)
        for i in range(self.num_agents):
            all_action_q = self.f_functions[i](torch.cat([*xi_s[i], si_s[i]], dim=2))
            all_action_q_for_agent[i, :, :] = all_action_q

            action_ids = action_vector[i].max(dim=2, keepdim=True)[1]
            action_q = all_action_q.gather(2, action_ids)
            curr_action_q_for_agent[i, :, :] = action_q

        # return_vec = []
        if ret_all_actions:
             return all_action_q_for_agent, curr_action_q_for_agent
        else:
            return curr_action_q_for_agent

        # if regularize:
        #     reg_vec = []
        #     for i in range(self.num_agents):
        #         attend_mag_reg = 1e-3 * sum((logit ** 2).mean() for logit in
        #                                     all_attend_logits[i])
        #         reg_vec.append(attend_mag_reg)
        #     return_vec.append(reg_vec)
        # return return_vec


def unit_test():
    batch_size = 20
    agents = 3
    act_size = 5
    obs_size = 10

    critic = Critic(observation_size=10, action_size=act_size, num_agents=agents, attention_heads=4, gpu=False)
    obs_vector = torch.randn((agents, batch_size, obs_size))
    action_vector = torch.randn((agents, batch_size, act_size))

    output = critic.forward(obs_vector, action_vector)
    assert(output.shape == (agents,batch_size,1))
    print("PASSED")
    
    output = critic.forward(obs_vector, action_vector, ret_all_actions=True)[0]
    assert(output.shape == (agents,batch_size,act_size))
    print("PASSED")

if __name__ == "__main__":
    unit_test()


