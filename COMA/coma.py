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

    def __init__(self, env, batch_size, seq_len, discount, n_agents, action_size, obs_size, state_size, h_size):
        """
        :param batch_size:
        :param seq_len: horizon
        :param discount: discounting factor
        :param n_agents:
        :param action_size: size of the discrete action space, actions are as one-hot vectors
        :param obs_size:
        :param state_size: size of global state
        :param h_size: size of GRU state
        """
        super(COMA, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.discount = discount
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

        # joint-action state pairs
        self.joint_action_state_pl = None

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

        self.critic = GlobalCritic(input_size=action_size*(n_agents) + state_size,
                         hidden_size=100)


    def fit_critic(self, lam):
        """
        Updates the critic using the off-line lambda return algorithm
        :param lam: lambda parameter used to average n-step targets
        :return:
        """

        # first compute the future discounted return at every step using the sequence of rewards

        G = np.zeros((self.batch_size, self.seq_len, self.seq_len + 1))
        # print('rewards', self.reward_seq_pl[0, :])

        # apply discount and sum
        total_return = self.reward_seq_pl.dot(np.fromfunction(lambda i: self.discount**i, shape=(self.seq_len,)))

        # print(total_return[0])

        # initialize the first column with estimates from the Q network
        predictions = self.critic.forward(self.joint_action_state_pl).squeeze()

        # use detach to assign to numpy array
        G[:, :, 0] = predictions.detach().numpy()

        # by moving backwards, construct the G matrix
        for t in range(self.seq_len - 1, -1, -1):

            # loop from one-step lookahead to pure MC estimate
            for n in range(1, self.seq_len + 1):

                # pure MC
                if t + n > self.seq_len - 1:
                    G[:, t, n] = total_return

                # combination of MC + bootstrapping
                else:
                    G[:, t, n] = self.reward_seq_pl[:, t] + self.discount*G[:, t+1, n-1]

        # compute target at timestep t
        targets = torch.zeros((self.batch_size, self.seq_len), dtype=torch.float32)

        # vector of powers of lambda
        weights = np.fromfunction(lambda i: lam**i, shape=(self.seq_len,))

        # normalize
        weights = weights * (1 - lam) / (1 - lam ** self.seq_len)

        # should be 1
        print(np.sum(weights))

        # Optimizer
        optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.005, eps=1e-08)

        for t in range(self.seq_len-1, -1, -1):
            targets[:, t] = torch.from_numpy(G[:, t, 1:].dot(weights))
            #print('target', targets[:, t])
            pred = self.critic.forward(self.joint_action_state_pl[:, t]).squeeze()
            #print('pred', pred)

            loss = torch.mean(torch.pow(targets[:, t] - pred, 2))
            print('loss', loss)

            # fit the Critic
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

                    # use the observation to construct global state
                    # the global state consists of positions + velocity of agents, first 4 elements from obs
                    self.global_state_pl[i, t, n*4:4*n+4] = obs_n[n][0:4]

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

                # get the absolute landmark positions for the global state
                self.global_state_pl[i, t, self.n_agents*4:] = np.array([landmark.state.p_pos for landmark in self.env.world.landmarks]).flatten()

        # concatenate the joint action, global state
        self.joint_action_state_pl = torch.from_numpy(np.concatenate((self.joint_action_pl, self.global_state_pl), axis=-1))
        self.joint_action_state_pl.requires_grad_(True)

def unit_test():
    n_agents = 3
    n_landmarks = 3

    test_coma = COMA(env=env, batch_size=1, seq_len=13, discount=0.99, n_agents=3, action_size=5, obs_size=14, state_size=18, h_size=16)
    test_coma.gather_rollouts(eps=0.05)
    print(test_coma.actor_input_pl[0].shape)
    print(test_coma.reward_seq_pl.shape)
    print(test_coma.joint_action_pl.shape)
    print(test_coma.global_state_pl.shape)
    print(test_coma.global_state_pl[0, 1, :])
    print(test_coma.joint_action_state_pl[0, 1, :])

    for e in range(20):
        test_coma.fit_critic(lam=0.1)

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

