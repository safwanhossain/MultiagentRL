#!/usr/bin/python3

import torch
import numpy as np
import torch.nn as nn

""" The Agent-dependent Critic.Each critic (one per agent) is a global critic and can access
all observations from each agent (along with their actions). There is an attention layer for each
critic that will pay special attention to whatever is deemed important
"""

class Global_Critic(torch.nn.Module):
    """
    Although we have multiple global critics, this can be efficiently computed with a single module
    There are shared layers for all critics, and layers specific to each global critic. As such,
    it will return a Q value for each agent.
    """
    def __init__(self, observation_size, action_size, num_agents):
        super(Global_Critic, self).__init__()
        self.num_agents = num_agents
        self.action_size = action_size
        self.observation_size = observation_size
        self.embedding_dim = 128
        
        # unique to each agent's Q function
        self.g_functions = []
        self.f_functions = []
        for i in range(self.num_agents):
            self.g_functions.append(nn.Sequential(
                nn.Linear(observation_size+action_size, self.embedding_dim),
                nn.ReLU()
            ))
            self.f_functions.append(nn.Sequential(
                nn.Linear(self.embedding_dim + 
                    self.embedding_dim*self.num_agents, self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, 1)
            ))

        # Shared across all critics 
        # Matrix mult with a vector can be represented as a single linear layer
        self.Wq_layers = [nn.Linear(self.embedding_dim, 
            self.embedding_dim) for i in range(self.num_agents)]
        self.Wk_layers = [nn.Linear(self.embedding_dim, 
            self.embedding_dim) for i in range(self.num_agents)]
        self.V_layers = [nn.Linear(self.embedding_dim, 
            self.embedding_dim) for i in range(self.num_agents)]
       
    def forward(self, observation_vector, action_vector):
        """
        :param state_action: joint action, global state, one-hot agent index
        :return: Q-value for the joint state
        """
        batch_size = observation_vector[0].shape[0]
        assert(len(observation_vector) == self.num_agents)
        
        # First compute the embedding
        ei_s =[]
        for i in range(len(observation_vector)):
            combined = torch.cat([observation_vector[i], action_vector[i]], dim=1)
            ei_s.append(self.g_functions[i](combined))
        print(ei_s[0].shape)

        xi_s = []
        for i in range(len(observation_vector)):    # for each x_i
            to_concat = []
            for l in range(self.num_agents):             # for each of the multiple attention heads
                query = self.Wq_layers[l](ei_s[i])
                total = torch.zeros(batch_size, self.embedding_dim) 
                for j in range(len(observation_vector)):
                    if i != j:
                        key = self.Wk_layers[l](ei_s[j])
                        alpha_j = torch.zeros(batch_size, 1)
                        for b in range(batch_size):
                            alpha_j[b,:] = torch.nn.functional.softmax(key[b,:].dot(query[b,:]))
                        assert(alpha_j.shape == (batch_size,1))
                        v_j = torch.nn.functional.leaky_relu(self.V_layers[l](ei_s[j]))
                        total += torch.mul(alpha_j,v_j)
                assert(total.shape == (batch_size, self.embedding_dim))
                to_concat.append(total)
            xi = torch.cat(to_concat, dim=1)
            assert(xi.shape == (batch_size, self.embedding_dim*self.num_agents))
            xi_s.append(xi)
        
        q_for_agent = []
        for i in range(self.num_agents):
            q_for_agent.append(self.f_functions[i](torch.cat([xi_s[i], ei_s[i]], dim=1)))

        return q_for_agent

def unit_test():
    critic = Global_Critic(observation_size=10, action_size=5, num_agents=3)
    obs_vector = [torch.randn((20, 10)) for i in range(3)]
    action_vector = [torch.randn((20, 5)) for i in range(3)]
    output = critic.forward(obs_vector, action_vector)
    
    print(output[0].shape)

if __name__ == "__main__":
    unit_test()


