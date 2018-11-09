import torch
import numpy as np
import torch.nn as nn

class Buffer():
    """ Buffer containing all the observations, actions and so an as an episode is played"""
    def __init__():
        # Setup the buffers which will store all the training information. The key for each buffer is the env id and the
        # data is a list where each index here represents a time step
        # Contains the joint values (joint observation, joint action, ...). Shared across all the parallel environments
        # and reset at the end of a sequence
        self.actions = {}
        self.rewards = {}
        self.agent_observations = {} 
        
        # Agent observation
        self.curr_agent_observations = {}
        self.next_agent_observations = {}

    def reset_all():
        self.actions = {}
        self.rewards = {}
        self.agent_observations = {} 
        self.curr_agent_observations = {}
        self.next_agent_observations = {}
        
    def is_empty():
        return (len(self.action_batch[0]) == 0)

    def length():
        return len(self.actions[0])

    def add_to_buffer(curr_obs_n, next_obs_n, joint_actions, reward, env_id):
        """" After every step, add the corresponding information to the buffers. Note that we have 
        a number of parallel environments. The buffers are stored according to their environment"""
        
        # Agent observation is it's own pos and velocity, pos of other agents wrt itself, and pos of
        # landmarks wrt itself
        curr_agent_obs, next_agent_obs = torch.zeros(self.n_agents, self.agent_obs_size), \
                torch.zeros(self.n_agents, self.agent_obs_size)
        for n in self.n_agents:
            curr_agent_obs[n] = curr_obs_n[n][0:self.agent_obs_size]
            next_agent_obs[n] = next_obs_n[n][0:self.agent_obs_size]

        self.curr_global_observations[env_id].append(curr_global_obs)
        self.next_global_observations[env_id].append(next_global_obs)
        
        self.curr_agent_observations[env_id].append(curr_agent_obs)
        self.next_agent_observations[env_id].append(next_agent_obs)
       
        joint_actions = joint_actions.view(self.n_agents, -1)
        self.actions[env_id].append(joint_actions)
        self.rewards[env_id].append(reward)
    
    def sample_from_buffer(batch_size): 
        """When we train the critic or actor, we need to sample a batch of observation, actions and
        so on from the buffer. We will sample batch_size/num_envs from each env.
        Returns a batch of samples each of: agent observations, actions, and rewards""" 

        # length of the buffer
        buffer_length = len(curr_global_obs[0])
        num_samples_per_env = batch_size / len(self.parallel_envs)
        assert(num_samples_per_env*len(self.parallel_envs) == batch_size)

        # Randomly choose the indicies to make up the batch
        indicies = np.random.choice(np.arange(buffer_length), size=num_samples_per_env, replace=True)
        
        # batches to return
        curr_agent_obs_batch = torch.zeros(len(self.parallel_envs), num_samples_per_env, self.n_agents, self.state_size)
        next_agent_obs_batch = torch.zeros(len(self.parallel_envs), num_samples_per_env, self.n_agenst, self.state_size)
        action_batch = torch.zeros(len(self.parallel_envs), num_samples_per_env, self.n_agents, self.action_size)
        reward_batch = torch.zeros(len(self.parallel_envs), num_samples_per_env, 1)

        # populate the batches
        for l in range(len(self.parallel_envs)):
            curr_agent_obs_batch[l,:] = self.curr_agent_observations[l][indicies]
            next_agent_obs_batch[l,:] = self.next_agent_observations[l][indicies]
            action_batch[l,:] = self.actions[l][indicies]
            reward_batch[l,:] = self.rewards[l][indicies]

        # reshape them to be correct dimensions
        curr_agent_obs_batch = curr_agent_obs_batch.view(self.n_agents, batch_size, -1)
        next_agent_obs_batch = next_agent_obs_batch.view(self.n_agents, batch_size, -1)
        action_batch = action_batch.view(self.n_agents, batch_size, -1)
        reward_batch = reward_batch.view(batch_size, 1)

        return curr_agent_obs_batch, next_agent_obs_batch, action_batch, reward_batch

