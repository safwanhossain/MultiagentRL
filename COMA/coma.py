#!/usr/bin/python3
from comet_ml import Experiment

from actors import GRUActor, MLPActor
from environment.particle_env import make_env, visualize
from critic import Critic
import numpy as np

from utils.buffer import Buffer
from utils.gather_batch import gather_batch

import torch
import time

""" Run the COMA training regime. Training is done in batch mode with a batch size of 30. Given the 
small nature of the current problem, we are NOT parameter sharing between agents (will do so when 
things start working). In each episode (similar to "epoch" in normal ML parlance), we:
    (1) collect sequences of data (batch size/n); 
    (2) use this batch of samples to compute the Value function; 
    (3) use the Value function and compute the baseline and update the actor and critic weights
    
"""

class COMA():

    def __init__(self, env, critic_arch, policy_arch, batch_size, seq_len, discount, lam, h_size,
                 lr_critic=0.0005, lr_actor=0.0002, eps=0.01):
        """
        :param batch_size:
        :param seq_len: horizon
        :param discount: discounting factor
        :param lam: for TD(lambda) return
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
        self.lam = lam
        self.n_agents = env.n
        self.action_size = env.action_size
        self.obs_size = env.agent_obs_size
        self.state_size = env.global_state_size
        self.h_size = h_size
        self.env = env
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.eps = eps
        self.epochs = 4000
        self.metrics = {}

        self.params = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "discount": discount,
            "n_agents": env.n,
            "h_size": h_size,
            "lambda": lam,
            "lr_critic": lr_critic,
            "lr_actor": lr_actor,
                }

        self.critic_arch = critic_arch
        self.policy_arch = policy_arch

        # The buffer to hold all the information of an episode
        self.buffer = Buffer(self.seq_len * self.batch_size, self.seq_len, self.batch_size * self.seq_len,
                             self.n_agents, env.agent_obs_size, env.global_state_size, env.action_size)

        # Create "placeholders" for incoming training data (Sorry, tensorflow habit)
        # will be set with process_data

        # joint-action state pairs
        self.joint_action_state_pl = torch.zeros((batch_size, seq_len, state_size+self.action_size*self.n_agents))

        # obs, prev_action pairs, one tensor for each agent
        self.actor_input_pl = [torch.zeros((batch_size, seq_len, obs_size+self.action_size+self.n_agents)) for n in range(self.n_agents)]

        # sequence of immediate rewards
        self.reward_seq_pl = np.zeros((batch_size, seq_len))

        # set up the modules for actor-critic based on specified arch
        # specify GRU or MLP policy
        ActorType = policy_arch['type']
        # actor takes in agent index as well
        self.actor = ActorType(input_size=obs_size + self.action_size + self.n_agents,
                           h_size=policy_arch['h_size'],
                           action_size = self.action_size)

        self.critic = Critic(input_size=self.action_size*(self.n_agents) + state_size,
                                   hidden_size=critic_arch['h_size'])#,
                                   # n_layers=critic_arch['n_layers'])

        self.target_critic = Critic(input_size=self.action_size*(self.n_agents) + state_size,
                                          hidden_size=critic_arch['h_size'])#,
                                          # n_layers=critic_arch['n_layers'])

    def update_target(self):
        """
        Updates the target network with the critic's weights
        :return:
        """
        self.target_critic.load_state_dict(self.critic.state_dict())

    def fit_actor(self, eps):
        """
        Updates the actor using policy gradient with counterfactual baseline.
        Accumulates actor gradients from each agent and performs the update for every time-step
        :return:
        """

        # first get the Q value for the joint actions
        # print('q input', self.joint_action_state_pl[0, :, :])

        q_vals = self.critic.forward(self.joint_action_state_pl)

        # print("q_vals", q_vals)

        diff_actions = torch.zeros_like(self.joint_action_state_pl)

        # initialize the advantage for each agent, by copying the Q values
        advantage = [q_vals.clone().detach() for a in range(self.n_agents)]

        # Optimizer
        optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor, eps=1e-08)
        optimizer.zero_grad()

        sum_loss = 0.0

        # computing baselines, by broadcasting across rollouts and time-steps
        for a in range(self.n_agents):

            # get the dist over actions, based on each agent's observation, use the same epsilon during fitting?
            action_dist = self.actor.forward(self.actor_input_pl[a], eps=eps)

            # make a copy of the joint-action, state, to substitute different actions in it
            diff_actions.copy_(self.joint_action_state_pl)

            # get the chosen action for each agent
            action_index = a * self.action_size
            chosen_action = self.joint_action_state_pl[:, :, action_index:action_index+self.action_size]

            # compute the baseline for that agent, by substituting different actions in the joint action
            for u in range(self.action_size):
                action = torch.zeros(self.action_size)
                action[u] = 1

                # index into that agent's action to substitute a different one
                diff_actions[:, :, action_index:action_index+self.action_size] = action

                # get the Q value of that new joint action
                Q_u = self.critic.forward(diff_actions)

                advantage[a] -= Q_u*action_dist[:, :, u].unsqueeze_(-1)

            # loss is negative log of probability of chosen action, scaled by the advantage
            # the advantage is treated as a scalar, so the gradient is computed only for log prob
            advantage[a] = advantage[a].detach()
            EPS = 1.0e-08
            # print('advantage', advantage[a].squeeze().size(), 'chosen_action', chosen_action.size())

            # sum along the action dim, left with log loss for chosen action
            loss = -torch.sum(torch.log(action_dist + EPS)*chosen_action, dim=-1)
            loss *= advantage[a].squeeze()

            print('a', a, 'action_dist', (action_dist[0, 0, :] + EPS))
            # print('action_dist', action_dist)
            print('advantage', torch.mean(advantage[a]))

            sum_loss += torch.mean(loss).item()

            # compute the gradients of the policy network using the advantage
            # do not use optimizer.zero_grad() since we want to accumulate the gradients for all agents
            loss.backward(torch.ones(self.batch_size, self.seq_len))

        # after computing the gradient for all agents, perform a weight update on the policy network
        optimizer.step()
        optimizer.zero_grad()
        self.metrics["mean_actor_loss"] = sum_loss / self.n_agents

        return sum_loss / self.n_agents


    def fit_critic(self):
        """
        Updates the critic using the off-line lambda return algorithm
        :return:
        """
        lam = self.lam

        # first compute the future discounted return at every step using the sequence of rewards

        G = np.zeros((self.batch_size, self.seq_len, self.seq_len + 1))
        # print('rewards', self.reward_seq_pl[0, :])

        # apply discount and sum
        total_return = self.reward_seq_pl.dot(np.fromfunction(lambda i: self.discount**i, shape=(self.seq_len,)))

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
                    G[:, t, n] = self.reward_seq_pl[:, t] + self.discount*G[:, t+1, n-1]

        # compute target at timestep t
        targets = torch.zeros((self.batch_size, self.seq_len), dtype=torch.float32)

        # Optimizer
        optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic, eps=1e-08)

        sum_loss = 0.

        for t in range(self.seq_len-1, -1, -1):

            # vector of powers of lambda
            weights = np.fromfunction(lambda i: lam ** i, shape=(self.seq_len - t,))

            # normalize
            weights = weights / np.sum(weights)

            print('t', t)
            targets[:, t] = torch.from_numpy(G[:, t, 1:self.seq_len-t+1].dot(weights))
            print('target', targets[0, t])
            pred = self.critic.forward(self.joint_action_state_pl[:, t]).squeeze()
            print('pred', pred[0])

            loss = torch.mean(torch.pow(targets[:, t] - pred, 2)) / self.seq_len
            sum_loss += loss.item()
            # print("critic loss", sum_loss)
            # fit the Critic
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.metrics["mean_critic_loss"] = sum_loss / self.seq_len
        return sum_loss / self.seq_len

    def format_buffer_data(self):
        """
        Reshape buffer data to correct dimensions
        """
        self.reward_seq_pl[:, :] = self.buffer.rewards[:, :].numpy()

        self.joint_action_state_pl[:, :, :self.state_size] = self.buffer.curr_global_state[:, :, :]
        self.joint_action_state_pl[:, :, self.state_size:] = self.buffer.actions.view(self.batch_size, self.seq_len, -1)

        noops = torch.zeros(self.batch_size, 1, self.n_agents, env.action_size)
        noops[:, 0, :, 0] = 1
        prev_action = torch.cat((noops, self.buffer.actions[:, :-1, :, :]), dim=1)

        for n in range(self.n_agents):
            agent_idx = torch.zeros(self.batch_size, self.seq_len, self.n_agents)
            agent_idx = agent_idx.scatter(2, torch.zeros(agent_idx.shape).fill_(n).long(), 1)

            actor_input = torch.cat((self.buffer.curr_agent_obs[:, :, n, :], prev_action[:, :, n, :], agent_idx), dim=2)
            actor_input = actor_input.view(self.batch_size, self.seq_len, -1).type(torch.FloatTensor)
            self.actor_input_pl[n][:, :, :] = actor_input

    def policy(self, obs, prev_action, n):
        agent_idx = torch.zeros(self.n_agents).scatter(0, torch.zeros(self.n_agents).fill_(n).long() , 1)
        actor_input = torch.cat((obs, prev_action, agent_idx)).view(1, 1, -1).type(torch.FloatTensor)
        return coma.actor(actor_input, eps=self.eps)[0][0]

    def train(self):
        experiment = Experiment(api_key='1jl4lQOnJsVdZR6oekS6WO5FI', project_name="COMA", \
                                auto_param_logging=False, auto_metric_logging=False,
                                log_graph=False, log_env_details=False, parse_args=False,
                                disabled=True)

        experiment.log_multiple_params(coma.params)
        experiment.log_multiple_params(coma.policy_arch)
        experiment.log_multiple_params(coma.critic_arch)

        for e in range(self.epochs):
            gather_batch(self.env, self)
            experiment.log_multiple_metrics(coma.metrics)
            experiment.set_step(e)

            self.format_buffer_data()

            self.fit_critic()
            self.fit_actor(eps=max(0.5 - e * 0.00005, 0.05))

            self.buffer.reset()

            if e % 2 == 0:
                print('e', e)
                self.update_target()


if __name__ == "__main__":

    n = 2
    obs_size = 4 + 2*(n-1) + 2*n
    state_size = 4*n + 2*n

    env = make_env(n_agents=n)

    policy_arch = {'type': GRUActor, 'h_size': 128}
    critic_arch = {'h_size': 64, 'n_layers':1}

    coma = COMA(env=env, critic_arch=critic_arch, policy_arch=policy_arch,
                batch_size=50, seq_len=50, discount=0.8, lam=0.8, h_size=128, lr_critic=0.0005, lr_actor=0.0002)




    coma.train()

    visualize(coma)