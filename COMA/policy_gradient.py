#!/usr/bin/python3

import torch
import numpy as np
import torch.nn as nn
from global_critic import GlobalCritic
from actor import MLPActor, GRUActor
import marl_env

""" Run the COMA training regime. Training is done in batch mode with a batch size of 30. Given the 
small nature of the current problem, we are NOT parameter sharing between agents (will do so when 
things start working). In each episode (similar to "epoch" in normal ML parlance), we:
    (1) collect sequences of data (batch size/n); 
    (2) use this batch of samples to compute the Value function; 
    (3) use the Value function and compute the baseline and update the actor and critic weights

"""


class PolicyGradient():

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
        self.joint_action_pl = torch.zeros((batch_size, seq_len, action_size * n_agents))

        # the global state
        self.global_state_pl = torch.zeros((batch_size, seq_len, state_size))

        # joint-action state pairs
        self.joint_action_state_pl = torch.zeros((batch_size, seq_len, state_size + self.action_size * self.n_agents))

        # obs, prev_action pairs, one tensor for each agent
        self.actor_input_pl = [torch.zeros((batch_size, seq_len, obs_size + action_size + n_agents)) for n in
                               range(self.n_agents)]

        # sequence of future returns for each timestep G[t] = r[t] + discount * G[t+1] used as target for Q-fitting
        self.return_seq_pl = np.zeros((batch_size, seq_len))

        # sequence of immediate rewards
        self.reward_seq_pl = np.zeros((batch_size, seq_len))

        # set up the modules for actor-critic
        # actor takes in agent index as well
        self.actor = GRUActor(input_size=obs_size + action_size + n_agents,
                              h_size=h_size,
                              action_size=action_size)

    def fit_actor(self, eps):
        """
        Updates the actor using policy gradient with counterfactual baseline.
        Accumulates actor gradients from each agent and performs the update for every time-step
        :return:
        """

        # apply discount and sum
        disc_rewards = self.reward_seq_pl.dot(np.fromfunction(lambda i: self.discount ** i, shape=(self.seq_len,)))
        future_return = np.triu(np.ones((self.seq_len, self.seq_len))).dot(disc_rewards)

        # Optimizer
        optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0005, eps=1e-08)
        optimizer.zero_grad()

        sum_loss = 0.0

        # computing baselines, by broadcasting across rollouts and time-steps
        for a in range(self.n_agents):

            # get the dist over actions, based on each agent's observation, use the same epsilon during fitting?
            action_dist = self.actor.forward(self.actor_input_pl[a], eps=eps)

            # get the chosen action for each agent
            action_index = a * self.action_size
            chosen_action = self.joint_action_state_pl[:, :, action_index:action_index + self.action_size]

            # loss is negative log of probability of chosen action, scaled by the return
            EPS = 1.0e-08
            # print('advantage', advantage[a].squeeze().size(), 'chosen_action', chosen_action.size())

            # sum along the action dim, left with log loss for chosen action
            loss = -torch.sum(torch.log(action_dist + EPS) * chosen_action, dim=-1)
            loss *= future_return.squeeze()

            print('a', a, 'action_dist', (action_dist[0, 0, :] + EPS))

            sum_loss += torch.mean(loss).item()
            # compute the gradients of the policy network using the advantage
            # do not use optimizer.zero_grad() since we want to accumulate the gradients for all agents
            loss.backward(torch.ones(self.batch_size, self.seq_len))

            # after computing the gradient for all agents, perform a weight update on the policy network
            optimizer.step()
            optimizer.zero_grad()

        return sum_loss

    def compute_return(self, lam):
        """
        Updates the critic using the off-line lambda return algorithm
        :param lam: lambda parameter used to average n-step targets
        :return:
        """

        # first compute the future discounted return at every step using the sequence of rewards

        G = np.zeros((self.batch_size, self.seq_len, self.seq_len + 1))
        # print('rewards', self.reward_seq_pl[0, :])

        # apply discount and sum
        total_return = self.reward_seq_pl.dot(np.fromfunction(lambda i: self.discount ** i, shape=(self.seq_len,)))

        # print(total_return[0])

        # initialize the first column with estimates from the target Q network
        predictions = self.target_critic.forward(self.joint_action_state_pl).squeeze()

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
                    G[:, t, n] = self.reward_seq_pl[:, t] + self.discount * G[:, t + 1, n - 1]

        # compute target at timestep t
        targets = torch.zeros((self.batch_size, self.seq_len), dtype=torch.float32)

        # Optimizer
        optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.0005, eps=1e-08)

        sum_loss = 0.

        for t in range(self.seq_len - 1, -1, -1):
            # vector of powers of lambda
            weights = np.fromfunction(lambda i: lam ** i, shape=(self.seq_len - t,))

            # normalize
            weights = weights / np.sum(weights)

            print('t', t)
            targets[:, t] = torch.from_numpy(G[:, t, 1:self.seq_len - t + 1].dot(weights))
            print('target', targets[0, t])
            pred = self.critic.forward(self.joint_action_state_pl[:, t]).squeeze()
            print('pred', pred[0])

            loss = torch.mean(torch.pow(targets[:, t] - pred, 2)) / self.seq_len
            sum_loss += loss.data[0]

            loss = torch.mean(torch.pow(targets[:, t] - pred, 2))
            sum_loss += loss.item()

            # print('loss', loss)

            # fit the Critic
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return sum_loss / self.seq_len


def unit_test():
    n_agents = 3
    n_landmarks = 3

    test_coma = COMA(env=env, batch_size=1, seq_len=30, discount=0.9, n_agents=3, action_size=5, obs_size=14,
                     state_size=18, h_size=16)
    test_coma.gather_rollouts(eps=0.05)
    print(test_coma.actor_input_pl[0].shape)
    print(test_coma.reward_seq_pl.shape)
    print(test_coma.joint_action_pl.shape)
    print(test_coma.global_state_pl.shape)
    print(test_coma.global_state_pl[0, 1, :])
    print(test_coma.joint_action_state_pl[0, 1, :])

    for e in range(20):
        test_coma.fit_actor(eps=0.05)

    # for e in range(20):
    #     test_coma.fit_critic(lam=0.5)


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

