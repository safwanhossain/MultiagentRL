import torch
import numpy as np

class Buffer():
    """ Buffer containing all the observations, actions and so an as an episode is played"""
    def __init__(self, batch_size, seq_len, n_agents, agent_obs_size, global_obs_size, action_size,
                 num_parallel_envs=1):
        """
        Setup the buffers which will store all the training information as entries containing (in order):
            - current agent observations (observations per agent at given timestep)
            - next agent observations (observations per agent at next timestep)
            - current global observations (global observations at given timestep)
            - next global observations (global observations at next timestep)
            - actions taken by each agents from current state
            - reward from environment at current state
        :param seq_len: length of an episode
        :param batch_size: batch size (only used when sampling from buffer)
        :param n_agents: number of agents
        :param agent_obs_size: size of observation for an agent
        :param global_obs_size: size of global observations
        :param action_size: number of actions (for one-hot encoding)
        :param num_parallel_envs: number of parallel environments
        """
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_agents = n_agents
        self.agent_obs_size = agent_obs_size
        self.global_obs_size = global_obs_size
        self.action_size = action_size

        # The buffer
        self.reset()

    def reset(self):
        """
        Empties the buffer and resets necessary parameters
        """
        self.curr_agent_obs = torch.zeros(self.batch_size, self.seq_len, self.n_agents, self.agent_obs_size)
        self.next_agent_obs = torch.zeros(self.batch_size, self.seq_len, self.n_agents, self.agent_obs_size)
        self.curr_global_state = torch.zeros(self.batch_size, self.seq_len, self.global_obs_size)
        self.next_global_state = torch.zeros(self.batch_size, self.seq_len, self.global_obs_size)
        self.actions = torch.zeros(self.batch_size, self.seq_len, self.n_agents, self.action_size)
        self.rewards = torch.zeros(self.batch_size, self.seq_len)
        self.end_index = torch.zeros(self.batch_size)


    def add_to_buffer(self, batch_index, t, curr_agent_obs, next_agent_obs, curr_global_state, next_global_state,
                      joint_actions, reward):
        """
        Add 1 timestep entry to buffer
        :param t: timestep index
        :param curr_agent_obs: agent obs
        :param next_agent_obs: next agent obs
        :param curr_global_state: state
        :param next_global_state: next state
        :param joint_actions: agent actions (one hot)
        :param reward: shared reward
        """
        self.curr_agent_obs[batch_index, t, :, :] = curr_agent_obs
        self.next_agent_obs[batch_index, t, :, :] = next_agent_obs
        self.curr_global_state[batch_index, t, :] = curr_global_state
        self.next_global_state[batch_index, t, :] = next_global_state
        self.actions[batch_index, t, :, :] = joint_actions
        self.rewards[batch_index, t] = reward

    def set_end_index(self, batch_index, t):
        """
        Set the index representing the last timestep for batch
        :param batch_index: Which batch this is the last timestep
        :param t: the last timestep index
        :return: None
        """
        self.end_index[batch_index] = t
