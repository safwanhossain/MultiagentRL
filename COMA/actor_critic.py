"""Implement COMA architecture for simple_spread environment N=3

Action space is discrete, 5 actions

Note: Agents have access to almost everything about the global state except for other agent's velocity
The GRU cell is still useful to model where other agents are going

Observation space:
Agent’s own velocity 2D
Agent’s own position 2D
Landmark positions with respect to the agent 3*2D
The positions of other agents with respect to the agent 2*2D
The messages C from other agents 2*2D messages (DISCARD)

"""

import torch
import numpy as np

class Actor(torch.nn.Module):

    def __init__(self, input_size, h_size, action_size):

        super(Actor, self).__init__()
        self.input_size = input_size
        self.h_size = h_size

        self.actor_gru = torch.nn.GRU(input_size=input_size,
                                      hidden_size=h_size,
                                      batch_first=True)

        self.linear = torch.nn.Linear(h_size, action_size)

    def forward(self, obs_seq, eps):
        """

        :param obs_seq: a sequence of shape (batch, seq_len, input_size)
        where input_size refers to size of [obs, prev_action]
        :param eps: softmax lower bound, for exploration
        :return:
        """
        batch_size = obs_seq.size()[0]
        h0 = torch.zeros(batch_size, 1, self.h_size)

        # output has shape [batch, seq_len, h_size]
        output, hn = self.actor_gru(obs_seq, h0)
        logits = self.linear(output)

        # compute eps-bounded softmax
        softmax = torch.nn.functional.softmax(logits, dim=2)
        return (1 - eps) * softmax + eps / self.action_size

class Critic(torch.nn.Module):

    def __init__(self, input_size, hidden_size, action_size):

        super(Critic, self).__init__()
        self.input_size = input_size
        self.output_size = action_size

        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, action_size)

    def forward(self, state_action):
        """

        :param state_action: joint action, global state, one-hot agent index
        :return: Q-values for each action
        """
        h = torch.nn.functional.tanh(self.linear1(state_action))
        return self.linear2(h)

class COMA():

    def __init__(self, batch_size, n_agents, action_size, obs_size, state_size, h_size):
        """

        :param batch_size:
        :param n_agents:
        :param action_size: size of the discrete action space, actions are as one-hot vectors
        :param obs_size:
        :param state_size: size of global state
        :param h_size: size of GRU state
        """
        super(COMA, self).__init__()
        self.batch_size = batch_size
        self.n_agents = n_agents
        self.action_size = action_size
        self.obs_size = obs_size
        self.state_size = state_size
        self.h_size = h_size

        # Create "placeholders" for incoming training data (Sorry, tensorflow habit)
        # will be set with process_data
        self.joint_fixed_actions_pl = None
        self.policy_input_pl = None
        self.global_state_pl = None
        self.return_seq_pl = None

        # set up the modules for actor-critic
        self.actor = Actor(input_size=obs_size + action_size,
                           h_size=h_size,
                           action_size = action_size)

        self.critic = Critic(input_size=action_size*(n_agents - 1) + state_size + n_agents,
                         action_size=action_size,
                         hidden_size=action_size*state_size)


    def process_data(self, joint_action, global_state, observations, reward_seq):
        """
        places numpy arrays of episode data into placeholders
        :param joint_action: [batch_size, seq_length, n_agents, action_size]
        :param global_state: [batch_size, seq_length, state_size]
        :param observations: [batch_size, seq_length, n_agents, obs_size]
        :param reward_seq: [batch_size, seq_length]
        :return:
        """
        batch_size = joint_action.shape()[0]
        seq_length = joint_action.shape()[1]
        assert(joint_action.shape() == (batch_size, seq_length, self.n_agents, self.action_size))
        assert(observations.shape() == (batch_size, seq_length, self.n_agents, self.obs_size))
        assert(global_state.shape() == (batch_size, seq_length, self.state_size))
        assert(reward_seq.shape() == (batch_size, seq_length))

        self.global_state_pl = global_state
        # process the data for Q and policy fitting

        # compute the future return at every timestep for each trajectory
        # use a lower triangular matrix to efficiently compute the sums
        L = np.tril(np.ones((seq_length, seq_length), dtype=int), -1)
        self.return_seq_pl = self.reward_seq.dot(L)

        # list of u_a_ for each agent
        self.joint_fixed_actions_pl = []

        # mask for indexing actions
        mask = np.ones(self.n_agents)
        for a in range(self.n_agents):
            mask[a] = 0
            # index all actions except the agent's action
            self.joint_fixed_actions_pl.append(joint_action[:, :, mask, :])
            mask[a] = 1

        # first prev_action is the zero action
        prev_actions = np.stack([np.zeros((batch_size, 1, self.n_agents, self.action_size)),
                                 joint_action[:, 0:-1, :, :]],
                                     axis=1)

        self.policy_input_pl = np.stack([observations, prev_actions], axis=-1)

    def train_coma(self, joint_action, global_state, observations, reward_seq):

        self.process_data(joint_action, global_state, observations, reward_seq)

        for a in range(self.n_agents):
            dist_actions
            advantage = self.critic.forward(self.policy_input_pl)


