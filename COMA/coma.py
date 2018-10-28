#!/usr/bin/python3

import torch
import numpy as np
import torch.nn as nn
from global_critic import GlobalCritic
from actor import Actor
import marl_env


""" Run the COMA training regime. Training is done in batch mode with a batch size of 30. Given the 
small nature of the current problem, we are NOT parameter sharing between agents (will do so when 
things start working). In each episode (similar to "epoch" in normal ML parlance), we:
    (1) collect sequences of data (batch size/n); 
    (2) use this batch of samples to compute the Value function; 
    (3) use the Value function and compute the baseline and update the actor and critic weights
    
"""

class COMA():

    def __init__(self, env, batch_size, seq_len, n_agents, action_size, obs_size, state_size, h_size):
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
        self.seq_len = seq_len
        self.n_agents = n_agents
        self.action_size = action_size
        self.obs_size = obs_size
        self.state_size = state_size
        self.h_size = h_size
        self.env = env

        # Create "placeholders" for incoming training data (Sorry, tensorflow habit)
        # will be set with process_data

        # list of u_- for each agent, the joint action excluding the agent's action, used for Q-fitting and baseline
        self.joint_fixed_actions_pl = None

        # joint action of all agents, flattened
        self.joint_action_pl = np.zeros((batch_size, seq_len, action_size*n_agents), dtype=np.float32)

        # the global state
        self.global_state_pl = np.zeros((batch_size, seq_len, state_size), dtype=np.float32)

        # obs, prev_action pairs, one tensor for each agent
        self.actor_input_pl = [np.zeros((batch_size, seq_len, obs_size+action_size), dtype=np.float32) for n in range(self.n_agents)]

        # sequence of future returns for each timestep G[t] = r[t] + discount * G[t+1] used as target for Q-fitting
        self.return_seq_pl = np.zeros((batch_size, seq_len), dtype=np.float32)

        # sequence of immediate rewards
        self.reward_seq_pl = np.zeros((batch_size, seq_len), dtype=np.float32)

        # set up the modules for actor-critic
        self.actor = Actor(input_size=obs_size + action_size,
                           h_size=h_size,
                           action_size = action_size)

        self.critic = GlobalCritic(input_size=action_size*(n_agents - 1) + state_size + n_agents,
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

        self.global_state_pl = torch.from_numpy(global_state)
        self.joint_action_pl = torch.from_numpy(joint_action)
        # process the data for Q and policy fitting

        # compute the future return at every timestep for each trajectory
        # use a lower triangular matrix to efficiently compute the sums
        L = np.tril(np.ones((seq_length, seq_length), dtype=int), -1)
        self.return_seq_pl = torch.from_numpy(self.reward_seq.dot(L))

        # first prev_action is the zero action NOOP
        prev_actions = np.stack([np.zeros((batch_size, 1, self.n_agents, self.action_size)),
                                 joint_action[:, 0:-1, :, :]],
                                     axis=1)

        self.policy_input_pl = torch.from_numpy(np.stack([observations, prev_actions], axis=-1))

    def gather_rollouts(self, eps):
        """
        gathers rollouts under the current policy
        :param eps:
        :return:
        """
        # Step 1: generate batch_size rollouts
        for i in range(self.batch_size):
            done = False
            joint_action = np.zeros((self.n_agents, self.action_size))
            env.reset()

            for t in range(self.seq_len):

                # get observations
                obs_n, reward_n, done_n, info_n = env.step(joint_action)

                # they all get the same reward, save the reward
                self.reward_seq_pl[i, t] = reward_n[0]

                # save the joint action
                self.joint_action_pl[i, t, :] = joint_action.flatten()

                # reset the joint action, one-hot representation
                joint_action = np.zeros((self.n_agents, self.action_size))

                # for each agent, save observation, compute next action
                for n in range(self.n_agents):

                    # get distribution over actions
                    obs_action = np.concatenate((obs_n[n][0:self.obs_size], joint_action[n, :]))
                    actor_input = torch.from_numpy(obs_action).view(1, 1, -1).type(torch.FloatTensor)

                    # save the actor input for training
                    self.actor_input_pl[n][i, t, :] = actor_input

                    pi = self.actor.forward(actor_input, eps)

                    # sample action from pi, convert to one-hot vector
                    action_idx = (torch.multinomial(pi[0, 0, :], num_samples=1)).numpy()
                    action = np.zeros(self.action_size)
                    action[action_idx] = 1
                    joint_action[n, :] = action


def unit_test():
    n_agents = 3
    n_landmarks = 3

    test_coma = COMA(env=env, batch_size=30, seq_len=2, n_agents=3, action_size=5, obs_size=14, state_size=18, h_size=16)
    test_coma.gather_rollouts(eps=0.05)
    print(test_coma.actor_input_pl[0].shape)
    print(test_coma.reward_seq_pl.shape)
    print(test_coma.joint_action_pl.shape)


if __name__ == "__main__":

        env = marl_env.make_env('simple_spread')

        """
        5 possible actions (NOOP, LEFT, RIGHT, UP, DOWN)
    
        Observation space:
        Agent’s own velocity 2D
        Agent’s own position 2D
        Landmark positions with respect to the agent 3*2D
        The positions of other agents with respect to the agent 2*2D
        The messages C from other agents 2*2D messages (DISCARD)
        
        Note: Agents have access to almost everything about the global state except for other agent's 
        velocity. The GRU cell is still useful to model where other agents are going '''
        """

        # global state consists of agent positions and velocity (2D) + landmark positions
        # state_size = n_agents*4 + n_landmarks*2

        unit_test()

