#!/usr/bin/python3

import torch
import numpy as np
import torch.nn as nn
from global_critic import GlobalCritic
from actor import *
import marl_env
from utils.base_model import BaseModel

""" Run the COMA training regime. Training is done in batch mode with a batch size of 30. Given the 
small nature of the current problem, we are NOT parameter sharing between agents (will do so when 
things start working). In each episode (similar to "epoch" in normal ML parlance), we:
    (1) collect sequences of data (batch size/n); 
    (2) use this batch of samples to compute the Value function; 
    (3) use the Value function and compute the baseline and update the actor and critic weights
    
"""

class COMA(BaseModel):

    def __init__(self, env, critic_arch, policy_arch, batch_size, seq_len, discount, lam, n_agents, action_size,
                 obs_size, state_size, lr_critic=0.0005, lr_actor=0.0002, alpha=0.1, **kwargs):
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
        self.gpu_mode = True
        super(COMA, self).__init__(use_gpu=self.gpu_mode)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.discount = discount
        self.lam = lam
        self.n_agents = n_agents
        self.action_size = action_size
        self.obs_size = obs_size
        self.state_size = state_size
        self.env = env
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.alpha = alpha
        self.metrics = {}
        self.set_flags(**kwargs)

        self.critic_arch = critic_arch
        self.policy_arch = policy_arch

        # Create "placeholders" for incoming training data (Sorry, tensorflow habit)
        # will be set with process_data

        # list of u_- for each agent, the joint action excluding the agent's action, used for Q-fitting and baseline
        self.joint_fixed_actions_pl = None

        # joint action of all agents, flattened
        self.joint_action_pl = torch.zeros((batch_size, seq_len, action_size*n_agents))

        # the global state
        self.global_state_pl = torch.zeros((batch_size, seq_len, state_size))

        # joint-action state pairs
        self.joint_action_state_pl = torch.zeros((batch_size, seq_len, state_size+self.action_size*self.n_agents))
        self.joint_action_state_pl = self.joint_action_state_pl.to(self.device)

        # obs, prev_action pairs, one tensor for each agent
        self.actor_input_pl = [torch.zeros((batch_size, seq_len, obs_size+action_size+n_agents)).to(self.device)
                               for _ in range(self.n_agents)]

        # sequence of future returns for each timestep G[t] = r[t] + discount * G[t+1] used as target for Q-fitting
        self.return_seq_pl = np.zeros((batch_size, seq_len))

        # sequence of immediate rewards
        self.reward_seq_pl = np.zeros((batch_size, seq_len))

        # set up the modules for actor-critic based on specified arch
        # specify GRU or MLP policy
        ActorType = policy_arch['type']
        # actor takes in agent index as well
        self.actor = ActorType(input_size=obs_size + action_size + n_agents,
                           h_size=policy_arch['h_size'],
                           device=self.device,
                           action_size = action_size)

        self.critic = GlobalCritic(input_size=action_size*(n_agents) + state_size,
                                   hidden_size=critic_arch['h_size'],
                                   device=self.device,
                                   n_layers=critic_arch['n_layers'])

        self.target_critic = GlobalCritic(input_size=action_size*(n_agents) + state_size,
                                          hidden_size=critic_arch['h_size'],
                                          device=self.device,
                                          n_layers=critic_arch['n_layers'])

        # actor Optimizer
        self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=self.lr_actor, eps=1e-08)

        # critic Optimizer
        self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr=self.lr_critic, eps=1e-08)

        # set fit_critic based on TD_LAMBDA flag
        if self.TD_LAMBDA:
            self.update_critic = self.td_lambda

        else:
            self.update_critic = self.td_one

        self.set_params()
        self.experiment.log_multiple_params(self.policy_arch)
        self.experiment.log_multiple_params(self.critic_arch)

        if isinstance(self.actor, GRUActor):
            AgentType = GRUAgent
        else:
            AgentType = MLPAgent

        self.agents = [AgentType(self.actor) for _ in range(self.n_agents)]

        if self.gpu_mode:
            cuda = torch.device('cuda')
            for agent in self.agents:
                agent.actor_net.cuda()
            self.critic.cuda()
            self.target_critic.cuda()

    def set_flags(self, **kwargs):
        """
        sets the flags for several training "features"
        :param kwargs:
        :return:
        """
        # whether to use soft-actor critic
        self.SAC = kwargs['SAC']

        # whether to use weighted n step returns or just one step
        self.TD_LAMBDA = kwargs['TD_LAMBDA']

        # TODO:
        # whether to use counterfactual baseline or v(s)
        #self.COMA_BASELINE = kwargs['COMA_BASELINE']

        # TODO:
        # whether to allow agents access to the global state
        #self.PARTIAL_OBS = kwargs['PARTIAL_OBS']

    def load_model(self, path, set_eval=True):

        self.actor = torch.load(path+"_actor.pt")
        self.critic = torch.load(path+"_critic")

        # if set_eval:
        #     self.critic.eval()
        #     self.actor.eval()

    def save_model(self):

        torch.save(self.critic.state_dict(), "saved_models/" + self.experiment.get_key() + "_critic")
        torch.save(self.actor.state_dict(), 'saved_models/' + self.experiment.get_key() + "_actor.pt")

    def reset_agents(self):
        for agent in self.agents:
            agent.reset_state()

    def update_target(self):
        """
        Updates the target network with the critic's weights
        :return:
        """
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update_actor(self):
        """
        Updates the actor using policy gradient with counterfactual baseline.
        Accumulates actor gradients from each agent and performs the update for every time-step
        :return:
        """

        # first get the Q value for the joint actions
        # print('q input', self.joint_action_state_pl[0, :, :])

        q_vals = self.critic.forward(self.joint_action_state_pl)

        # print("q_vals", q_vals)

        diff_actions = torch.zeros_like(self.joint_action_state_pl).to(self.device)

        # initialize the advantage for each agent, by copying the Q values
        advantage = [q_vals.clone().detach() for a in range(self.n_agents) ]

        self.actor_optimizer.zero_grad()

        sum_loss = 0.0

        # reset the hidden state of agents
        self.reset_agents()

        # computing baselines, by broadcasting across rollouts and time-steps
        for a in range(self.n_agents):

            self.agents[a].reset_state()

            # get the dist over actions, based on each agent's observation, don't use epsilon while fitting
            action_dist = self.agents[a].get_action_dist(self.actor_input_pl[a], eps=0)

            # make a copy of the joint-action, state, to substitute different actions in it
            diff_actions.copy_(self.joint_action_state_pl)

            # get the chosen action at the next time step for each agent
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

            # if using soft actor critic, add the policy's entropy to the advantage
            if self.SAC:
                # add entropy term
                entropy = torch.sum(torch.log(action_dist)*action_dist, dim=-1).unsqueeze_(-1)
                advantage[a] -= self.alpha * entropy

            # loss is negative log of probability of chosen action, scaled by the advantage
            # the advantage is treated as a scalar, so the gradient is computed only for log prob
            advantage[a] = advantage[a].detach()
            EPS = 1.0e-08
            # print('advantage', advantage[a].squeeze().size(), 'chosen_action', chosen_action.size())

            # sum along the action dim, left with log loss for chosen action
            loss = -torch.sum(torch.log(action_dist + EPS)*chosen_action, dim=-1)
            loss *= advantage[a].squeeze()

            #print('a', a, 'action_dist', (action_dist[0, 0, :] + EPS))
            # print('action_dist', action_dist)
            #print('advantage', torch.mean(advantage[a]))

            sum_loss += torch.mean(loss).item()

            # compute the gradients of the policy network using the advantage
            # do not use optimizer.zero_grad() since we want to accumulate the gradients for all agents
            loss.backward(torch.ones(self.batch_size, self.seq_len).to(self.device))

        # after computing the gradient for all agents, perform a weight update on the policy network
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()
        self.metrics["mean_actor_loss"] = sum_loss / self.n_agents

        return sum_loss / self.n_agents

    def td_one(self):
        """
        updates the critic using td(1)
        :return:
        """

        # create G with two columns, one for current prediction, second for one step lookahead
        G = np.zeros((self.batch_size, self.seq_len, 2))

        # initialize the first column with estimates from the target Q network
        predictions = self.target_critic.forward(self.joint_action_state_pl).squeeze()

        # use detach to assign to numpy array
        G[:, :, 0] = predictions.detach().cpu().numpy()

        sum_loss = 0.
        # by moving backwards, construct the G matrix
        for t in range(self.seq_len - 1, -1, -1):
            G[:, t, 1] = self.reward_seq_pl[:, t] + self.discount * G[:, t + 1, 0]

            # if using soft actor critic, compute the joint action dist for the next time step
            if self.SAC:
                joint_action_dist = 1
                for a in range(self.n_agents):
                    joint_action_dist *= self.agents[a].get_action_dist(self.actor_input_pl[a][:, t + 1, :], eps=0)

                G[:, t, 1] -= self.alpha * torch.sum(torch.log(joint_action_dist) * joint_action_dist, dim=-1)

            pred = self.critic.forward(self.joint_action_state_pl[:, t]).squeeze()

            loss = torch.mean(torch.pow(G[:, t, 1] - pred, 2)) / self.seq_len
            loss.to(self.device)
            sum_loss += loss.item()
            # print("critic loss", sum_loss)
            # fit the Critic
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

        self.metrics["mean_critic_loss"] = sum_loss / self.seq_len
        return sum_loss / self.seq_len


    def td_lambda(self):
        """
        Updates the critic using the off-line lambda return algorithm
        :return:
        """
        lam = self.lam

        # first compute the future discounted return at every step using the sequence of rewards
        G = np.zeros((self.batch_size, self.seq_len, self.seq_len))

        # initialize the first column with estimates from the target Q network
        predictions = self.target_critic.forward(self.joint_action_state_pl).squeeze()

        # use detach to assign to numpy array
        G[:, :, 0] = predictions.detach().cpu().numpy()

        # by moving backwards, construct the G matrix
        for t in range(self.seq_len - 1, -1, -1):

            # loop from one-step lookahead to pure MC estimate
            for n in range(1, self.seq_len - 1):

                # pure MC
                if t + n > self.seq_len - 1:
                    G[:, t, n] = self.reward_seq_pl[:, t:].dot(np.fromfunction(lambda i: self.discount**i,
                                                                                 shape=(self.seq_len-t,)))

                # combination of MC + bootstrapping
                else:
                    G[:, t, n] = self.reward_seq_pl[:, t] + self.discount*G[:, t+1, n-1]

        # compute target at timestep t
        targets = torch.zeros((self.batch_size, self.seq_len), dtype=torch.float32).to(self.device)

        sum_loss = 0.

        # by moving backwards except for the last transition, compute targets and update critic
        for t in range(self.seq_len-2, -1, -1):

            # vector of powers of lambda
            weights = np.fromfunction(lambda i: lam ** i, shape=(self.seq_len -1 - t,))

            # normalize
            weights = weights / np.sum(weights)

            #print('t', t)
            targets[:, t] = torch.from_numpy(G[:, t, 1:self.seq_len - t].dot(weights))

            # if using soft actor critic, compute the joint action dist for the next time step
            if self.SAC:
                joint_action_dist = 1
                for a in range(self.n_agents):
                    joint_action_dist *= self.agents[a].get_action_dist(self.actor_input_pl[a][:, t+1, :], eps=0)
                targets[:, t] -= self.alpha*torch.sum(torch.log(joint_action_dist) * joint_action_dist, dim=-1)

            #print('target', targets[0, t])
            pred = self.critic.forward(self.joint_action_state_pl[:, t]).squeeze()
            #print('pred', pred[0])

            loss = torch.mean(torch.pow(targets[:, t] - pred, 2)) / self.seq_len
            sum_loss += loss.item()
            # print("critic loss", sum_loss)
            # fit the Critic
            self.critic_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.critic_optimizer.step()

        self.metrics["mean_critic_loss"] = sum_loss / self.seq_len
        return sum_loss / self.seq_len

    def update(self, epoch):
        """
        update model
        """

        if epoch % 2 == 0:
            # print('e', epoch)
            self.update_target()

        critic_loss = self.update_critic()
        actor_loss = self.update_actor()

        return critic_loss, actor_loss



