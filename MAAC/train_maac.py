#!/usr/bin/python3

import torch
import numpy as np
import torch.nn as nn
from critic import Global_Critic
from actor import Actor_Policy
from episode_buffer import Buffer
import marl_env
from datatime import datetime

""" Run the MAAC training regime. Here we have E parallel environments, and a global critic for
each agent with an attention mechanism. For both the critic and actor loss, they use an entropy
term to encourage exploration. They also use a baseline to reduce variance."""

# In MAAC, we need to run a number of parallel environments. Set them up here
def make_parallel_environments(env_name, num_environments)
    # create multiple environments each with a different seed
    parallel_envs = []
    for i in range(num_environments):
        env = marl_env.make_env('simple_spread')
        env.seed(i*1000)
        parallel_envs.append(env)
    return parallel_envs


class MAAC():
    def __init__(self, parallel_envs, n_agents, action_size, agent_obs_size):
        """
        :param batch_size:
        :param seq_len: horizon
        :param discount: discounting factor
        :param n_agents:
        :param action_size: size of the discrete action space, actions are as one-hot vectors
        :param obs_size:
        :param state_size: size of global state (position and velocity of each agent + location of landmarks)
        """
        
        super(MAAC, self).__init__()
        self.parallel_envs = parallel_envs
        self.batch_size = 1024
        self.seq_len = 100
        self.gamma = 0.95 
        self.n_agents = n_agents
        self.action_size = action_size
        self.obs_size = obs_size
        self.gpu_mode = True
        self.sequence_length = 25
        self.episodes = 10000
        self.batch_size = 64
        self.lr = 0.01
        self.tau = 0.002
        self.attend_tau = 0.04
        self.steps_per_update
        # The buffer to hold all the information of an episode
        self.buffer = Buffer() 
        
        # MAAC does NOT do parameter sharing - each agent has it's own network and optimizer
        self.agents = [Actor_Policy(input_size=self.obs_size, action_size=self.action_size) for i in range(self.n_agents)]
        self.target_agents = [Actor_Policy(input_size=self.obs_size, action_size=self.action_size) for i in range(self.n_agents)]
        self.agent_optimizers = [Adam(agent.get_params(), lr=self.lr) for agent in self.agents]
        
        # We usually have two versions of the critic, as TD lambda is trained thru bootstraping. self.critic
        # is the "True" critic and target critic is the one used for training
        self.critic = GlobalCritic(input_size=action_size*(n_agents) + state_size, hidden_size=100)
        self.target_critic = GlobalCritic(input_size=action_size*(n_agents) + state_size, hidden_size=100)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr, weight_decay=1e-3) 
        
        # get the critic values for all agents
        if self.gpu_mode:
            for agent in self.agents:
                agent.cuda()
            self.critic.cuda()
            self.target_critic.cuda()

    def update_critic(batch_size)
        """ train the critic usinng batches of observatios and actions. Training is based on the 
        TD lambda method with a target critic"""
        assert(not self.buffer().is_empty())

        # collect the batch
        curr_agent_obs_batch, next_agent_obs_batch, \
                action_batch, reward_batch = self.buffer.sample_from_buffer(batch_size)
        
        # get the critic values for all agents
        critic_values = self.critic(curr_obs_batch, action_batch)
        
        # Sample a batch of actions given the next observation (for the target network)
        agent_probs = torch.zeros(self.n_agents, self.action_size)
        next_joint_actions = torch.zeros(self.n_agents, self.action_size)
        for n in self.n_agents:
            probs = self.target_agents[n].action(next_agent_obs_batch[n])
            agent_probs[n] = probs
            action_idx = (torch.multinomial(probs, num_samples=1)).numpy().flatten()
            next_joint_actions[n][action_idx[l]] = 1

        # get the target critic values for all agents
        target_values = self.target_critic(next_obs_batch, next_joint_actions)

        # compute the critic loss
        critic_loss = 0
        for n in self.n_agents:
            y_i = reward_batch + self.gamma*(target_values[n] - self.alpha*torch.log(agent_probs[n]))
            critic_loss += torch.nn.MSELoss(critic_values[n] - y_i)
        
        #backpropagate the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    
    def update_agent(self.batch_size):
        """ Train the actor using the Q function to approximate long term reward, use a baseline to
        reduce variance, and use an entropy term to encourage exploration"""
        assert(not self.buffer().is_empty())
        
        # first sample a batch of observations from the buffer
        curr_obs_batch, next_obs_batch, curr_agent_obs_batch, next_agent_obs_batch, \
                action_batch, reward_batch = self.buffer.sample_from_buffer(batch_size)

        # Use the current observations to sample a set of actions (for the target network)
        agent_probs = torch.zeros(self.n_agents, self.action_size)
        next_joint_actions = torch.zeros(self.n_agents, self.action_size)
        for n in self.n_agents:
            probs = self.agents[n].action(curr_agent_obs_batch[n])
            agent_probs[n] = probs
            action_idx = (torch.multinomial(probs, num_samples=1)).numpy().flatten()
            next_joint_actions[n][action_idx[l]] = 1
        
        # Get the Q values for each agent at the current observation
        all_action_q, curr_action_q = self.critic(curr_obs_batch, action_batch, ret_all_actions=True)

        # compute the loss using baseline and entropy term
        for n in self.n_agents:
            log_pi = torch.log(agent_probs[n])
            baseline = (all_action_q[n]*agent_probs[n]).sum(dim=1, keepdim=True)
            target = curr_action_q[n] - baseline
            target = (target - target.mean()) / target.std()    # make it 0 mean and 1 var (idk why??)
            loss = (log_pi*(self.alpha*log_pi - target)).mean()

            # TODO: there is supposed to be some regularization here according to the github code
            # but not the paper. Not implemented
            
            self.agent_optimizers[n].zero_grad()
            loss.backward()
            self.agent_optimizers[n].step()

    
    def update_target_networks():
        """ The target networks should basically mimic the trained networks with some 1-tau probability"""
        # First update the non-attention modules for critic
        for target_param, param in zip(self.target_critic.get_non_attention_parameters(),\
                self.critic.non_attention_parameters()):
            target_param.data.copy_(target_param.data * (self.tau) + param.data * (1 - self.tau))
        
        # Then update the attention modules for critic
        for target_param, param in zip(self.target_critic.get_attention_parameters(),\
                self.critic.get_attention_parameters()):
            target_param.data.copy_(target_param.data * self.attend_tau + param.data * (1 - self.attend_tau))
        
        # Now update the target agents
        for n in self.n_agents:
            for target_param, param in zip(self.target_agents[n].get_params(), self.agents[n].get_params()):
                target_param.data.copy_(target_param.data * self.tau + param.data * (1 - self.attend_tau))


    def train():
        num_steps = 0
        for ep in range(0, self.episodes):
            # Step 1: initialize actions to NOOP and reset environments
            joint_actions = torch.zeros(len(maac.parallel_envs), maac.n_agents, maac.action_size)
            curr_obs_n = []
            reward_arr = torch.zeros(len(maac.parallel_envs), seq_len)
            # for some reason, they do not clear the buffer - I find this very strange

            for l in range(len(maax.parallel_envs)):
                joint_actions[l, :, 0] = 1
            for env in maac.parallel_envs:
                prev_obs_n.append(env.reset())

            for i in range(self.seq_length):
                # Need to store actor observations to get the next action
                obs_for_actor = torch.zeros(self.n_agents, len(self.parallel_envs), self.obs_size)
                
                for e, env in enumerate(self.parallel_envs):
                    # get observations, by executing current joint action and store in buffer
                    next_obs_n, reward_n, done_n, info_n = env.step(joint_actions[e])
                    reward = reward_n[0]
                    reward_arr[e,i] = reward
                    add_to_buffer(curr_obs_n, next_obs_n, joint_actions[e], reward, e):
                    
                    # store the observations needed for actor forward pass
                    for n in range(self.n_agents):
                        obs_for_actor[n,e,:] = torch.from_numpy(obs_n[n][0:self.obs_size])  

                    # next observation becomes the current ones
                    curr_obs_n[e] = next_obs_n
                    num_steps += len(self.parallel_envs)

                # Step 3: compute the next action
                joint_actions = torch.zeros(len(maac.parallel_envs), maac.n_agents, maac.action_size)
                for agent in self.agents:
                    obs = obs_for_actor[n]
                    dist = agent.action(obs)
                    
                    # sample action from pi, convert to one-hot vector
                    action_idx = (torch.multinomial(dist, num_samples=1)).numpy().flatten()
                    for l in len(maac.parallel_envs):
                        joint_actions[l][n][action_idx[l]] = 1

                if self.buffer.length() > self.batch_size and num_steps >= self.steps_per_update:
                    for i range(self.num_updates):
                        update_critic()
                        update_agent()
                    update_target_networks()
                    num_steps = 0

            # print the reward at end of each episode
            print('reward', torch.mean(reward_arr))

def main():
    parallel_envs = make_parallel_environments("simple_spread", 12)
    big_MAAC = MAAC(parallel_envs, n_agents=3, action_size=5, agent_obs_size=14)    
    big_MAAC.train()


