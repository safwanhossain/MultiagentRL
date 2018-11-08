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
    def __init__(self, observation_size, action_size, num_agents, attention_heads):
        super(Global_Critic, self).__init__()
        self.num_agents = num_agents
        self.action_size = action_size
        self.observation_size = observation_size
        self.attention_heads = attention_heads
        self.embedding_dim = 128
        self.attend_dim = self.embedding_dim // attention_heads

        # unique to each agent's Q function
        self.g_functions = []
        self.f_functions = []
        for i in range(self.num_agents):
            self.g_functions.append(nn.Sequential(
                nn.Linear(observation_size+action_size, self.embedding_dim),
                nn.LeakyReLU()
            ))
            self.f_functions.append(nn.Sequential(
                nn.Linear(self.embedding_dim + 
                    self.attend_dim*self.attention_heads, self.embedding_dim),
                nn.LeakyReLU(),
                nn.Linear(self.embedding_dim, 1)
            ))

        # Shared across all critics 
        # Matrix mult with a vector can be represented as a single linear layer
        self.Wq_layers = [nn.Linear(self.embedding_dim, 
            self.attend_dim, bias=False) for i in range(self.attention_heads)]
        self.Wk_layers = [nn.Linear(self.embedding_dim, 
            self.attend_dim, bias=False) for i in range(self.attention_heads)]
        self.V_layers = [nn.Linear(self.embedding_dim, 
            self.attend_dim) for i in range(self.attention_heads)]
        weight_init(mean=0.0, std=0.02)
    
    def weight_init(self, mean, std):
        for m in self._modules:
            utils.normal_init(self._modules[m], mean, std)
       
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

        xi_s = []
        for i in range(len(observation_vector)):    # for each x_i
            to_concat = []
            for l in range(self.attention_heads):             # for each of the multiple attention heads
                query = self.Wq_layers[l](ei_s[i])
                total = torch.zeros(batch_size, self.attend_dim) 
                for j in range(len(observation_vector)):
                    if i != j:
                        key = self.Wk_layers[l](ei_s[j])
                        alpha_j = torch.zeros(batch_size, 1)
                        for b in range(batch_size):
                            scaled_att_weights = (key[b,:].dot(query[b,:])) / np.sqrt(key.shape[1])
                            alpha_j[b,:] = torch.nn.functional.softmax(scaled_att_weights, dim=0)
                        assert(alpha_j.shape == (batch_size,1))
                        v_j = torch.nn.functional.leaky_relu(self.V_layers[l](ei_s[j]))
                        total += torch.mul(alpha_j,v_j)
                assert(total.shape == (batch_size, self.attend_dim))
                to_concat.append(total)
            xi = torch.cat(to_concat, dim=1)
            assert(xi.shape == (batch_size, self.embedding_dim))
            xi_s.append(xi)
        
        q_for_agent = []
        for i in range(self.num_agents):
            q_for_agent.append(self.f_functions[i](torch.cat([xi_s[i], ei_s[i]], dim=1)))

        return q_for_agent

def unit_test():
    batch_size = 20
    agents = 3

    critic = Global_Critic(observation_size=10, action_size=5, num_agents=agents, attention_heads=4)
    obs_vector = [torch.randn((batch_size, 10)) for i in range(agents)]
    action_vector = [torch.randn((batch_size, 5)) for i in range(agents)]
    output = critic.forward(obs_vector, action_vector)
    
    assert(len(output) == agents)
    assert(output[0].shape == (batch_size,1))
    print("PASSED")


if __name__ == "__main__":
    unit_test()


