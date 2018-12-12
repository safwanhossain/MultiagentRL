import torch
import numpy as np

class Buffer():
    """ Buffer containing all the observations, actions and so an as an episode is played"""
    def __init__(self, max_size, seq_len, batch_size, n_agents, agent_obs_size, global_obs_size, action_size,
                 num_parallel_envs=1):
        """
        Setup the buffers which will store all the training information as entries containing (in order):
            - current agent observations (observations per agent at given timestep)
            - next agent observations (observations per agent at next timestep)
            - current global observations (global observations at given timestep)
            - next global observations (global observations at next timestep)
            - actions taken by each agents from current state
            - reward from environment at current state
        :param max_size: maximum size of the buffer
        :param seq_len: length of an episode
        :param batch_size: batch size (only used when sampling from buffer)
        :param n_agents: number of agents
        :param agent_obs_size: size of observation for an agent
        :param global_obs_size: size of global observations
        :param action_size: number of actions (for one-hot encoding)
        :param num_parallel_envs: number of parallel environments
        """
        # size of buffer in terms of transition tuples
        # max_size = max_episodes * seq_len
        self.max_size = max_size
        self.seq_len = seq_len
        self.n_agents = n_agents
        self.agent_obs_size = agent_obs_size
        self.global_obs_size = global_obs_size
        self.action_size = action_size

        # max num of episodes that contribute to the buffer data
        self.max_episodes = max_size // seq_len

        # num_parallel_envs from which we sample episodes
        self.num_parallel_envs = num_parallel_envs

        # size of training batch (in terms of tuples?)
        self.batch_size = batch_size

        self.full = False

        self.samples_per_episode = 0

        # The buffer index is set to 0
        # Note that the buffer index points to the distinct episodes saved in the buffer,
        # it does the same thing as episode index (first index)
        self.reset()


    def reset(self):
        """
        Empties the buffer and resets necessary parameters
        """

        # data is idexed by episode index and time step
        self.curr_agent_obs_pl = torch.zeros(self.max_episodes, self.seq_len, self.n_agents, self.agent_obs_size)
        self.next_agent_obs_pl = torch.zeros(self.max_episodes, self.seq_len, self.n_agents, self.agent_obs_size)
        self.curr_global_state_pl = torch.zeros(self.max_episodes, self.seq_len, self.global_obs_size)
        self.next_global_state_pl = torch.zeros(self.max_episodes, self.seq_len, self.global_obs_size)
        self.actions_pl = torch.zeros(self.max_episodes, self.seq_len, self.n_agents, self.action_size)
        self.rewards_pl = torch.zeros(self.max_episodes, self.seq_len)
        self.samples_per_episode = 0
        self.buffer_index = 0
        self.full = False

        # data placeholders compatible with training
        self.joint_action_state_pl = torch.zeros(
            (self.max_episodes, self.seq_len, self.global_obs_size + self.action_size * self.n_agents))

        # the global state
        self.global_state_pl = torch.zeros((self.max_episodes, self.seq_len, self.global_obs_size))

        # joint action of all agents, flattened
        self.joint_action_pl = torch.zeros((self.max_episodes, self.seq_len, self.action_size * self.n_agents))

        # obs, prev_action pairs, one tensor for each agent
        self.actor_input_pl = \
            [torch.zeros((self.max_episodes, self.seq_len, self.agent_obs_size + self.action_size + self.n_agents))
             for _ in range(self.n_agents)]

        self.agent_observations = torch.zeros(self.n_agents, self.max_episodes, self.seq_len, self.agent_obs_size)
        self.agent_actions = torch.zeros(self.n_agents, self.max_episodes, self.seq_len, self.action_size)

        # sequence of immediate rewards
        self.reward_seq_pl = np.zeros((self.max_episodes, self.seq_len))

    def is_empty(self):
        """
        Whether or not the buffer contains data
        :return: True if buffer contains data, else false
        """

        # buffer index can be set to 0 whenever buffer is full but not officially emptied yet
        return (not self.full) and (self.buffer_index == 0)

    def length(self):
        """
        Number of timesteps (episodes * seq len) stored in the buffer
        :return: Number of entries in buffer
        """
        return self.max_episodes * self.seq_len if self.full else self.buffer_index * self.seq_len

    def add_to_buffer(self, t, curr_agent_obs, next_agent_obs, curr_global_state, next_global_state,
                      joint_actions, reward, env_id=None):
        """
        Add 1 timestep entry to buffer
        :param t: timestep index
        :param curr_agent_obs: agent obs
        :param next_agent_obs: next agent obs
        :param curr_global_state: state
        :param next_global_state: next state
        :param joint_actions: agent actions (one hot)
        :param reward: shared reward
        :param env_id: Necessary if running parallel environments
        """
        if self.num_parallel_envs != 1 and env_id is None:
            raise TypeError("Environment id is required if using parallel environments")

        # takes into account episodes generated by parallel environments using env_id
        # compatible with experience replay since self.max_episodes is max_size // seq_len
        episode_index = (self.buffer_index + (env_id or 0)) % self.max_episodes

        self.curr_agent_obs_pl[episode_index, t, :, :] = curr_agent_obs
        self.next_agent_obs_pl[episode_index, t, :, :] = next_agent_obs
        self.curr_global_state_pl[episode_index, t, :] = curr_global_state
        self.next_global_state_pl[episode_index, t, :] = next_global_state
        self.actions_pl[episode_index, t, :, :] = joint_actions
        self.rewards_pl[episode_index, t] = reward

        # if we reached the end of an episode, increment the buffer index
        if t == self.seq_len - 1:
            self.buffer_index += self.num_parallel_envs
            if not self.full and self.batch_size % self.buffer_index == 0:
                self.samples_per_episode = self.batch_size // self.buffer_index

            # support for circular buffer, in case of experience replay
            if self.buffer_index > self.max_episodes:
                self.buffer_index = 0
                self.full = True

    def sample_from_buffer(self, num_samples, use_gpu=True):

        """
        NOTE: unused before experience replay
        Sample n samples from buffer.
        :return: data of n episodes
        """
        # generate random indices to sample episodes
        episode_i = np.random.choice(self.max_episodes, size=num_samples, replace=False)

        sampled_data = {"joint_action_state": self.joint_action_state_pl[episode_i, :, :],
                        "actor_input": [self.actor_input_pl[n][episode_i, :, :] for n in range(self.n_agents)],
                        "reward_seq": self.reward_seq_pl[episode_i, :],
                        "observations": self.agent_observations[:, episode_i, :, :],
                        "actions": self.agent_actions[:, episode_i, :, :]}

        return sampled_data

    def format_buffer_data(self):
        """
        Reshape buffer data to correct dimensions for maac
        """
        self.reward_seq_pl[:, :] = self.rewards_pl[:, :].numpy()

        self.joint_action_state_pl[:, :, :self.action_size * self.n_agents] = self.actions_pl.view(self.max_episodes,
                                                                                                       self.seq_len, -1)
        # pair actions with the state at which they were taken
        self.joint_action_state_pl[:, :, self.action_size * self.n_agents:] = self.curr_global_state_pl[:, :, :]

        # observations are one step ahead of actions here, because they will be paired and fed to the actor network
        self.agent_observations[:, :, :, :] = self.next_agent_obs_pl.permute(2, 0, 1, 3)
        self.agent_actions[:, :, :, :] = self.actions_pl.permute(2, 0, 1, 3)

        for n in range(self.n_agents):
            agent_idx = torch.zeros(self.max_episodes, self.seq_len, self.n_agents)
            agent_idx = agent_idx.scatter(2, torch.zeros(agent_idx.shape).fill_(n).long(), 1)

            actor_input = torch.cat(
                (self.next_agent_obs_pl[:, :, n, :], self.actions_pl[:, :, n, :], agent_idx), dim=2)
            actor_input = actor_input.view(self.max_episodes, self.seq_len, -1).type(torch.FloatTensor)
            self.actor_input_pl[n][:, :, :] = actor_input

        return None