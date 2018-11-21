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
                nn.BatchNorm1d(observation_size+action_size),
                nn.Linear(observation_size+action_size, self.embedding_dim),
                nn.LeakyReLU()
            ))
            self.state_only_embeddings.append(nn.Sequential(
                nn.BatchNorm1d(observation_size),
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
       
    def forward(self, observation_vector, action_vector, ret_all_actions=False):
        """
        observation vector is the observation for each agent. We are NOT using some global
        state representations since it can't really work well with this attention model
        """
        batch_size = observation_vector.shape[1]
        assert(observation_vector.shape == (self.num_agents, batch_size, self.observation_size))
        assert(action_vector.shape == (self.num_agents, batch_size, self.action_size))
        
        # First compute the embedding
        ei_s =[]
        si_s =[]
        for i in range(self.num_agents):
            combined = torch.cat([observation_vector[i], action_vector[i]], dim=1)
            ei_s.append(self.g_functions[i](combined))
            si_s.append(self.state_only_embeddings[i](observation_vector[i]))
        
        xi_s = []
        for i in range(self.num_agents):    # for each x_i
            to_concat = []
            for l in range(self.attention_heads):             # for each of the multiple attention heads
                query = self.Wq_layers[l](ei_s[i])
                total = torch.zeros(batch_size, self.attend_dim).to(self.device)
                for j in range(len(observation_vector)):
                    if i != j:
                        key = self.Wk_layers[l](ei_s[j])
                        alpha_j = torch.bmm(key.view(batch_size, 1, self.attend_dim), \
                                query.view(batch_size, self.attend_dim, 1)).view(-1,1)
                        alpha_j = alpha_j / np.sqrt(key.shape[1])
                        assert(alpha_j.shape == (batch_size,1))
                        v_j = torch.nn.functional.leaky_relu(self.V_layers[l](ei_s[j]))
                        total += torch.mul(alpha_j,v_j)
                assert(total.shape == (batch_size, self.attend_dim))
                to_concat.append(total)
            xi = torch.cat(to_concat, dim=1)
            assert(xi.shape == (batch_size, self.embedding_dim))
            xi_s.append(xi)
        
        all_action_q_for_agent = torch.zeros(self.num_agents, batch_size, self.action_size).to(self.device)
        curr_action_q_for_agent = torch.zeros(self.num_agents, batch_size, 1).to(self.device)
        for i in range(self.num_agents):
            all_action_q = self.f_functions[i](torch.cat([xi_s[i], si_s[i]], dim=1))
            all_action_q_for_agent[i,:,:] = all_action_q

            action_ids = action_vector[i].max(dim=1, keepdim=True)[1]
            action_q = all_action_q.gather(1, action_ids)
            curr_action_q_for_agent[i,:,:] = action_q

        if ret_all_actions:
            return all_action_q_for_agent, curr_action_q_for_agent
        else:
            return curr_action_q_for_agent


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


