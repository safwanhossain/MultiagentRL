#!/usr/bin/python3

import torch
import numpy as np
import torch.nn as nn
from critic import Global_Critic
from actor import Actor_Policy
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
    def __init__(self, parallel_envs, batch_size, seq_len, discount, n_agents, action_size, agent_obs_size, state_size):
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
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.discount = discount
        self.n_agents = n_agents
        self.action_size = action_size
        self.obs_size = obs_size
        self.gpu_mode = True
        self.sequence_length = 25
        self.batch_size = 64

        # Setup the buffers which will store all the training information. Each index here represents a time step
        # Contains the joint values (joint observation, joint action, ...). Shared across all the parallel environments
        # and reset at the end of a sequence
        self.actions = {}
        self.rewards = {}
        self.agent_observations = {} 
        
        # Global observations consists of location and velocity of all agent + location of landmarks 
        self.curr_global_observations = {}
        self.next_global_observations = {}

        # Agent observation
        self.curr_agent_observations = {}
        self.next_agent_observations = {}

        # set up the modules for actor-critic
        self.actor = Actor_Policy(input_size=self.obs_size, action_size=self.action_size)
        
        # We usually have two versions of the critic, as TD lambda is trained thru bootstraping. self.critic
        # is the "True" critic and target critic is the one used for training
        self.critic = GlobalCritic(input_size=action_size*(n_agents) + state_size, hidden_size=100)
        self.target_critic = GlobalCritic(input_size=action_size*(n_agents) + state_size, hidden_size=100)
        
        if self.gpu_mode:
            self.actor.cuda()
            self.critic.cuda()
            self.target_critic.cuda()


    def add_to_buffer(curr_obs_n, next_obs_n, joint_actions, reward, env_id):
        """" After every step, add the corresponding information to the buffers. Note that we have 
        a number of parallel environments. The buffers are stored according to their environment"""
        
        # Global observations consists of: pos and velocity of all agents + landmark position
        curr_global_obs, next_global_obs = torch.zeros(self.state_size), torch.zeros(self.state_size)
        # Agent observation is it's own pos and velocity, pos of other agents wrt itself, and pos of
        # landmarks wrt itself
        curr_agent_obs, next_agent_obs = torch.zeros(self.n_agents, self.agent_obs_size), \
                torch.zeros(self.n_agents, self.agent_obs_size)
        for n in self.n_agents:
            curr_global_obs[4*n:4*(n+1)] = curr_obs_n[n][0:4]
            next_global_obs[4*n:4*(n+1)] = next_obs_n[n][0:4]
            curr_agent_obs[n] = curr_obs_n[n][0:self.agent_obs_size]
            next_agent_obs[n] = next_obs_n[n][0:self.agent_obs_size]

        curr_global_obs[self.n_agents*4] = torch.from_numpy(np.array(
            [landmark.state.p_pos for landmark in self.parallel_envs[env_id].world.landmarks]).flatten())
        next_global_obs[self.n_agents*4] = torch.from_numpy(np.array(
            [landmark.state.p_pos for landmark in self.parallel_envs[env_id].world.landmarks]).flatten())
    
        self.curr_global_observations[env_id].append(curr_global_obs)
        self.next_global_observations[env_id].append(next_global_obs)
        
        self.curr_agent_observations[env_id].append(curr_agent_obs)
        self.next_agent_observations[env_id].append(next_agent_obs)
        
        self.actions[env_id] = joint_actions
        self.rewards[env_id] = reward


    def sample_from_buffer(batch_size): 
        """When we train the critic or actor, we need to sample a batch of observation, actions and
        so on from the buffer. We will sample batch_size/num_envs from each env.
        Returns a batch of samples each of: global and agent observations, actions, and rewards""" 

        # length of the buffer
        buffer_length = len(curr_global_obs[0])
        num_samples_per_env = batch_size / len(self.parallel_envs)
        assert(num_samples_per_env*len(self.parallel_envs) == batch_size)

        # Randomly choose the indicies to make up the batch
        indicies = np.random.choice(np.arange(buffer_length), size=num_samples_per_env, replace=False)
        
        # batches to return
        curr_obs_batch = torch.zeros(len(self.parallel_envs), num_samples_per_env, self.state_size)
        next_obs_batch = torch.zeros(len(self.parallel_envs), num_samples_per_env, self.state_size)
        curr_agent_obs_batch = torch.zeros(len(self.parallel_envs), num_samples_per_env, self.n_agents, self.state_size)
        next_agent_obs_batch = torch.zeros(len(self.parallel_envs), num_samples_per_env, self.n_agenst, self.state_size)
        action_batch = torch.zeros(len(self.parallel_envs), num_samples_per_env, self.n_agents*self.action_size)
        reward_batch = torch.zeros(len(self.parallel_envs), num_samples_per_env, 1)

        # populate the batches
        for l in range(len(self.parallel_envs)):
            curr_obs_batch[l,:] = self.curr_global_observations[l][indicies]
            next_obs_batch[l,:] = self.next_global_observations[l][indicies]
            curr_agent_obs_batch[l,:] = self.curr_agent_observations[l][indicies]
            next_agent_obs_batch[l,:] = self.next_agent_observations[l][indicies]
            action_batch[l,:] = self.actions[l][indicies]
            reward_batch[l,:] = self.rewards[l][indicies]

        return curr_obs_batch, next_obs_batch, curr_agent_obs_batch, next_agent_obs_batch, action_batch, reward_batch
       

    def update_critic()
        """ train the critic usinng batches of observatios and actions. Training is based on the 
        TD lambda method with a target critic"""
        
        # collect the batch
        assert(len(self.action_batch[0]) != 0)
        curr_obs_batch, next_obs_batch, curr_agent_obs_batch, next_agent_obs_batch, \
                action_batch, reward_batch = sample_from_buffer(self.batch_size)
       
        critic_values = self.critic(curr_obs_batch, action_batch)
        
        # Sample a batch of actions given the next observation (for the target network)
        agent_probs = torch.zeros(self.n_agents, self.action_size)
        for n in self.n_agents:
            probs = self.actor.action(next_agent_obs_batch[n])
            
            agent_probs[n] = probs

            













