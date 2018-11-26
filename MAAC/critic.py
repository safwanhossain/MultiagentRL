
#!/usr/bin/python3

import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import utils
import marl_env
import matplotlib.pyplot as plt
from pdb import set_trace as bp

""" The Agent-dependent Critic.Each critic (one per agent) is a global critic and can access
all observations from each agent (along with their actions). There is an attention layer for each
critic that will pay special attention to whatever is deemed important. In forward pass, the 
observations for all agents are passed in
"""

class Global_Critic(nn.Module):
    """
    Although we have multiple global critics, this can be efficiently computed with a single module
    There are shared layers for all critics, and layers specific to each global critic. As such,
    it will return a Q value for each agent.
    """
    def __init__(self, observation_size, action_size, num_agents, attention_heads, gpu=True):
        super(Global_Critic, self).__init__()
        self.num_agents = num_agents
        self.gpu = gpu
        self.action_size = action_size
        self.observation_size = observation_size
        self.attention_heads = attention_heads
        self.embedding_dim = 128
        self.attend_dim = self.embedding_dim // attention_heads

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
        #self.weight_init(mean=0.0, std=0.02)
    
    def get_non_attention_parameters(self):
        return (p for n, p in self.named_parameters() if 'layers' in n)
    
    def get_attention_parameters(self):
        return (p for n, p in self.named_parameters() if 'functions' in n or 'embeddings' in n)

    def weight_init(self, mean, std):
        for m in self._modules:
            utils.normal_init(self._modules[m], mean, std)
       
    def forward(self, observation_vector, action_vector, regularize=False, ret_all_actions=False):
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
        all_attend_logits = [[] for _ in range(self.num_agents)]
        
        for i in range(self.num_agents):    # for each x_i
            to_concat = []
            for l in range(self.attention_heads):             # for each of the multiple attention heads
                query = self.Wq_layers[l](ei_s[i])
                total = torch.zeros(batch_size, self.attend_dim) 
                if self.gpu:
                    total = total.cuda() 
                
                attend_logits_tensor = torch.zeros((2, batch_size, 1))
                for j in range(self.num_agents):
                    index = 0
                    if i != j:
                        key = self.Wk_layers[l](ei_s[j])
                        alpha_j = torch.bmm(key.view(batch_size, 1, self.attend_dim), \
                                query.view(batch_size, self.attend_dim, 1)).view(-1,1)
                        attend_logits_tensor[index,:,:] = alpha_j
                        alpha_j = torch.nn.functional.softmax(alpha_j / np.sqrt(key.shape[1]))
                        assert(alpha_j.shape == (batch_size,1))
                        v_j = torch.nn.functional.leaky_relu(self.V_layers[l](ei_s[j]))
                        total += torch.mul(alpha_j,v_j)
                        index += 1
                assert(total.shape == (batch_size, self.attend_dim))
                all_attend_logits[i].append(attend_logits_tensor.view(batch_size, 1, 2))
                to_concat.append(total)
            xi = torch.cat(to_concat, dim=1)
            assert(xi.shape == (batch_size, self.embedding_dim))
            xi_s.append(xi)
        
        all_action_q_for_agent = torch.zeros(self.num_agents, batch_size, self.action_size)
        curr_action_q_for_agent = torch.zeros(self.num_agents, batch_size, 1)
        for i in range(self.num_agents):
            all_action_q = self.f_functions[i](torch.cat([xi_s[i], si_s[i]], dim=1))
            all_action_q_for_agent[i,:,:] = all_action_q

            action_ids = action_vector[i].max(dim=1, keepdim=True)[1]
            action_q = all_action_q.gather(1, action_ids)
            curr_action_q_for_agent[i,:,:] = action_q

        return_vec = []
        if ret_all_actions:
            return_vec = [all_action_q_for_agent, curr_action_q_for_agent]
        else:
            return_vec= [curr_action_q_for_agent]
        
        if regularize:
            reg_vec = []
            for i in range(self.num_agents):
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                        all_attend_logits[i])
                reg_vec.append(attend_mag_reg)
            return_vec.append(reg_vec)
        return return_vec

def unit_test():
    batch_size = 20
    agents = 3
    act_size = 5
    obs_size = 10

    critic = Global_Critic(observation_size=10, action_size=act_size, num_agents=agents, attention_heads=4, gpu=False)
    obs_vector = torch.randn((agents, batch_size, obs_size))
    action_vector = torch.randn((agents, batch_size, act_size))

    output = critic.forward(obs_vector, action_vector)
    assert(output.shape == (agents,batch_size,1))
    print("PASSED")
    
    output = critic.forward(obs_vector, action_vector, ret_all_actions=True)[0]
    assert(output.shape == (agents,batch_size,act_size))
    print("PASSED")


def test_critic():
    """ We are going to test if the critic is being trained properly. To do this, we start off with 
    some random policy and keep this constant. From the environment, we know what the true q values
    should be. """
    
    # Make an environment
    env = marl_env.make_env('simple_spread')
    action_size = 5
    num_agents = 3
    policy_size = 2048
    agent_obs_size = 14

    # Generate a random policy 
    action_vector = torch.FloatTensor(num_agents, policy_size, action_size)
    action_vector.zero_()
    for i in range(num_agents):
        act = torch.LongTensor(policy_size,1).random_() % action_size
        action_vector[i].scatter_(1, act, 1)
    
    action_vector = action_vector.view(policy_size, num_agents, action_size)
    obs_array = torch.FloatTensor(policy_size, num_agents, agent_obs_size)
    reward_array = torch.FloatTensor(policy_size, 1)
    
    # Get the observation and reward for the policy
    reset_obs = env.reset()
    for j in range(num_agents):
        obs_array[0][j] = torch.FloatTensor(reset_obs[j][:agent_obs_size])

    for i in range(policy_size):
        next_obs_n, reward_n, done_n, info_n = env.step(action_vector[i])
        if i != policy_size-1:
            for j in range(num_agents):
                obs_array[i+1][j] = torch.FloatTensor(next_obs_n[j][:agent_obs_size])
        else:
            last_obs = next_obs_n
        reward_array[i] = reward_n[0]

    # Get the next observation and reward vector
    next_action_vector = action_vector.view(policy_size, num_agents, action_size)
    next_action_vector[0:-1] = action_vector[1:]
    next_action_vector[-1] = action_vector[0]
    
    next_obs_array = torch.FloatTensor(policy_size, num_agents, agent_obs_size)
    next_obs_array[0:-1] = obs_array[1:]
    for j in range(num_agents):
        next_obs_array[-1][j] = torch.FloatTensor(last_obs[j][:agent_obs_size])

    # Compute the true Q value for each obs/action 
    gamma = 0.95
    q_array = []
    for i in range(policy_size):
        powers = [i for i in range(0, policy_size-i)]
        constants = np.power(gamma, powers)
        q_val = np.sum(reward_array[i:].numpy()*constants)
        q_array.append(q_val)

    critic = Global_Critic(observation_size=agent_obs_size, \
            action_size=action_size, num_agents=num_agents, attention_heads=4).cuda()
    target_critic = Global_Critic(observation_size=agent_obs_size, \
            action_size=action_size, num_agents=num_agents, attention_heads=4).cuda()
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.0005) 

    epochs = 1500
    batch_size = 2000
    dataset = torch.utils.data.TensorDataset(obs_array, next_obs_array, action_vector, next_action_vector, reward_array)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) 
    print("Begin Critic training")

    avg_critic_losses = []
    for epoch in range(epochs):
        all_critic_losses = 0
        for obs, next_obs, action, next_action, reward in dataloader:
            obs = obs.view(num_agents, batch_size, agent_obs_size)
            next_obs = next_obs.view(num_agents, batch_size, agent_obs_size)
            action = action.view(num_agents, batch_size, action_size)
            next_action = action.view(num_agents, batch_size, action_size)
            
            critic_values = critic(obs.cuda(), action.cuda())
            target_values = target_critic(next_obs.cuda(), next_action.cuda()).detach()

            # compute the critic loss
            critic_loss = 0
            mse_loss = torch.nn.MSELoss().cuda()
            for n in range(num_agents):
                y_i = reward.cuda() + gamma*(target_values[n].cuda())
                critic_loss += mse_loss(critic_values[n].cuda(), y_i.detach())
            
            all_critic_losses += critic_loss

            # backpropagate the loss
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

        avg_critic_loss = float(all_critic_losses.cpu()[0]/1)
        print("The loss for epoch ", epoch, " is ", avg_critic_loss)
        avg_critic_losses.append(avg_critic_loss)

        """ The target networks should basically mimic the trained networks with some 1-tau probability"""
        tau = 0.002
        attend_tau = 0.04
        # First update the non-attention modules for critic
        for target_param, param in zip(target_critic.get_non_attention_parameters(),\
                critic.get_non_attention_parameters()):
            target_param.data.copy_(target_param.data * (tau) + param.data * (1 - tau))
        
        # Then update the attention modules for critic
        for target_param, param in zip(target_critic.get_attention_parameters(),\
                critic.get_attention_parameters()):
            target_param.data.copy_(target_param.data * (attend_tau) + param.data * (1 - attend_tau))
    
    plt.plot(np.arange(0,epochs,1), avg_critic_losses)
    plt.show()

if __name__ == "__main__":
    #unit_test()
    test_critic()

