import torch
import numpy as np

class Buffer():
    """ Buffer containing all the observations, actions and so an as an episode is played"""
    def __init__(self, max_size, seq_len, batch_size, n_agents, agent_obs_size, global_obs_size, action_size,
                 num_parallel_envs=1):
        """
        Setup the buffers which will store all the training information in the following order:
            - current agent observations (observations per agent at given timestep)
            - next agent observations (observations per agent at next timestep)
            - current global observations (global observations at given timestep)
            - next global observations (global observations at next timestep)
            - actions taken by each agents from current state
            - reward from environment at current state
        :param max_size: maximum size of the buffer
        :param n_agents: number of agents
        :param agent_obs_size: size of observation for an agent
        :param global_obs_size: size of global observations
        :param action_size: number of actions (for one-hot encoding)
        :param num_parallel_envs: number of parallel environments
        """
        self.max_size = max_size
        self.seq_len = seq_len
        self.n_agents = n_agents
        self.agent_obs_size = agent_obs_size
        self.global_obs_size = global_obs_size
        self.action_size = action_size
        self.max_episodes = max_size // seq_len
        self.num_parallel_envs = num_parallel_envs
        self.batch_size = batch_size
        self.full = False
        self.samples_per_episode = 0

        # The buffer
        self.reset()

    def reset(self):
        self.curr_agent_obs = torch.zeros(self.max_episodes, self.seq_len, self.n_agents, self.agent_obs_size)
        self.next_agent_obs = torch.zeros(self.max_episodes, self.seq_len, self.n_agents, self.agent_obs_size)
        self.curr_global_state = torch.zeros(self.max_episodes, self.seq_len, self.global_obs_size)
        self.next_global_state = torch.zeros(self.max_episodes, self.seq_len, self.global_obs_size)
        self.actions = torch.zeros(self.max_episodes, self.seq_len, self.n_agents, self.action_size)
        self.rewards = torch.zeros(self.max_episodes, self.seq_len)
        self.buffer_index = 0
        self.full = False

    def is_empty(self):
        return (not self.full) and (self.buffer_index == 0)

    def length(self):
        return self.max_episodes * self.seq_len if self.full else self.buffer_index * self.seq_len

    def add_to_buffer(self, t, curr_agent_obs, next_agent_obs, curr_global_state, next_global_state,
                      joint_actions, reward, env_id=None):
        """
        Add 1 timestep to buffer
        """
        if self.num_parallel_envs != 1 and env_id is None:
            raise TypeError("Environment id is required if using parallel environments")

        episode_index = (self.buffer_index + (env_id or 0)) % self.max_episodes

        self.curr_agent_obs[episode_index, t, :, :] = curr_agent_obs
        self.next_agent_obs[episode_index, t, :, :] = next_agent_obs
        self.curr_global_state[episode_index, t, :] = curr_global_state
        self.next_global_state[episode_index, t, :] = next_global_state
        self.actions[episode_index, t, :, :] = joint_actions
        self.rewards[episode_index, t] = reward

        if t == self.seq_len - 1:
            self.buffer_index += self.num_parallel_envs
            if not self.full and self.batch_size % self.buffer_index == 0:
                self.samples_per_episode = self.batch_size // self.buffer_index
            if self.buffer_index > self.max_episodes:
                self.buffer_index = 0
                self.full = True

    def sample_from_buffer(self, ordered, full_episode, use_gpu=True):
        """
        Sample a batch from buffer. Samples batch_size/self.num_envs from each env
        :param batch_size: Size of batch
        :param ordered: If true, ordered by increasing time, if false, shuffled.
        :return: Batch of data
        """
        batch_size = self.batch_size
        buffer_size = self.max_episodes if self.full else self.buffer_index
        samples_per_episode = self.seq_len if full_episode else self.samples_per_episode
        num_episodes = batch_size // samples_per_episode
        assert(batch_size % samples_per_episode == 0)
        assert(num_episodes * samples_per_episode == batch_size)

        episode_i = np.random.choice(buffer_size, size=num_episodes, replace=False)
        if ordered:
            timestep_i = range(samples_per_episode)

        # batches to return
        curr_agent_obs_batch = torch.zeros(num_episodes, samples_per_episode, self.n_agents, self.agent_obs_size)
        next_agent_obs_batch = torch.zeros(num_episodes, samples_per_episode, self.n_agents, self.agent_obs_size)
        curr_global_state_batch = torch.zeros(num_episodes, samples_per_episode, self.global_obs_size)
        next_global_state_batch = torch.zeros(num_episodes, samples_per_episode, self.global_obs_size)
        action_batch = torch.zeros(num_episodes, samples_per_episode, self.n_agents, self.action_size)
        reward_batch = torch.zeros(num_episodes, samples_per_episode)

        for i, e_i in enumerate(episode_i):
            if not ordered:
                # Randomly choose the indices to make up the batch
                timestep_i = np.random.choice(self.seq_len, size=samples_per_episode, replace=True).tolist()
            curr_agent_obs_batch[i, :, :, :] = self.curr_agent_obs[e_i, timestep_i, :, :]
            next_agent_obs_batch[i, :, :, :] = self.next_agent_obs[e_i, timestep_i, :, :]
            curr_global_state_batch[i, :, :] = self.curr_global_state[e_i, timestep_i, :]
            next_global_state_batch[i, :, :] = self.next_global_state[e_i, timestep_i, :]
            action_batch[i, :, :, :] = self.actions[e_i, timestep_i, :, :]
            reward_batch[i, :] = self.rewards[e_i, timestep_i]

        ret = (curr_agent_obs_batch, next_agent_obs_batch, curr_global_state_batch, next_global_state_batch,
               action_batch, reward_batch)

        #TODO add cuda
        if use_gpu:
            ret = [data.cuda() for data in ret]
        return ret

