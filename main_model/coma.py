#!/usr/bin/python3
from main_model.base_model import BaseModel
from main_model.actors import *
from main_model.critic import Critic
import numpy as np
import torch


""" Run the main_model training regime. Training is done in batch mode with a batch size of 30. Given the 
small nature of the current problem, we are NOT parameter sharing between agents (will do so when 
things start working). In each episode (similar to "epoch" in normal ML parlance), we:
    (1) collect sequences of data (batch size/n); 
    (2) use this batch of samples to compute the Value function; 
    (3) use the Value function and compute the baseline and update the actor and critic weights
    
"""

class COMA(BaseModel):

    def __init__(self, flags, envs, critic_arch, policy_arch, batch_size, replay_memory,
                 seq_len, discount, lam, lr_critic=0.0001, lr_actor=0.0001, alpha=0.1, log_files=None):
        """
        Initialize all aspects of the model
        :param env: Environment model will be used in
        :param batch_size:
        :param seq_len: horizon
        :param discount: discounting factor
        :param lam: for TD(lambda) return
        :param h_size: size of GRU state
        """
        use_gpu = flags.gpu.lower() in ["true", "t", "yes", "y"]
        track_results = flags.track_results.lower() in ["true", "t", "yes", "y"]
        use_gpu = flags.gpu.lower() in ["true", "t", "yes", "y"]
        track_results = flags.track_results.lower() in ["true", "t", "yes", "y"]
        super(COMA, self).__init__(envs, batch_size, replay_memory, seq_len, discount, lam,
                                    lr_critic=lr_critic, lr_actor=lr_actor, alpha=0.1, use_gpu=use_gpu,
                                    track_results=track_results, log_files=log_files)

        self.use_maac = False
        self.SAC = flags.SAC.lower() in ["true", "t", "yes", "y"]
        self.TD_LAMBDA = flags.TD_LAMBDA.lower() in ["true", "t", "yes", "y"]

        self.critic_arch = critic_arch
        self.policy_arch = policy_arch

        # set up the modules for actor-critic based on specified arch
        # specify GRU or MLP policy
        ActorType = policy_arch['type']
        # actor takes in agent index as well
        self.actor = ActorType(input_size=self.obs_size + self.action_size + self.n_agents,
                               h_size=policy_arch['h_size'],
                               action_size = self.action_size,
                               device=self.device)

        self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=self.lr_actor, eps=1e-08)

        self.critic = Critic(input_size=self.action_size * (self.n_agents) + self.state_size,
                              hidden_size=critic_arch['h_size'],
                              device=self.device,
                              n_layers=critic_arch['n_layers'],
                              num_agents=self.n_agents)

        self.target_critic = Critic(input_size=self.action_size*(self.n_agents) + self.state_size,
                                    hidden_size=critic_arch['h_size'],
                                    n_layers=critic_arch['n_layers'],
                                    device=self.device,
                                    num_agents=self.n_agents)

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

        self.params["lam"] = self.lam
        self.set_params()
        self.experiment.log_multiple_params(self.policy_arch)
        self.experiment.log_multiple_params(self.critic_arch)

    def reset_agents(self):
        for agent in self.agents:
            agent.reset_state()

    def update_target_network(self):
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
        # reset the hidden state of agents
        self.reset_agents()

        # first get the Q value for the joint actions
        # print('q input', self.joint_action_state_pl[0, :, :])

        q_vals = self.critic.forward(self.joint_action_state_pl).detach()

        # print("q_vals", q_vals)
        diff_action_in = self.joint_action_state_pl
        diff_actions = torch.zeros_like(diff_action_in).to(self.device)

        # initialize the advantage for each agent, by copying the Q values
        advantage = [q_vals.clone().detach() for a in range(self.n_agents)]

        # Optimizer
        self.actor_optimizer.zero_grad()

        sum_loss = 0.0
        EPS = 1.0e-08

        # computing baselines, by broadcasting across rollouts and time-steps
        for a in range(self.n_agents):
            # get the dist over actions, based on each agent's observation, use the same epsilon during fitting?
            pi = self.agents[a].get_action_dist(self.actor_input_pl[a], eps=0)

            # make a copy of the joint-action, state, to substitute different actions in it
            diff_actions.copy_(diff_action_in)

            # get the chosen action for each agent
            action_index = a * self.action_size
            chosen_action = self.joint_action_state_pl[:, :, action_index:action_index+self.action_size]

            # compute the baseline for that agent, by substituting different actions in the joint action
            for u in range(self.action_size):
                action = torch.zeros(self.action_size)
                action[u] = 1
                diff_actions[:, :, action_index:action_index+self.action_size] = action
                critic_in = diff_actions

                # get the Q value of that new joint action
                Q_u = self.critic(critic_in)
                advantage[a] -= Q_u*pi[:, :, u].unsqueeze_(-1)

            if self.SAC:
                entropy = torch.sum(torch.log(pi) * pi, dim=-1).unsqueeze_(-1)
                advantage[a] -= self.alpha * entropy

            # loss is negative log of probability of chosen action, scaled by the advantage
            # the advantage is treated as a scalar, so the gradient is computed only for log prob
            advantage[a] = advantage[a].detach()

            # sum along the action dim, left with log loss for chosen action
            loss = -torch.sum(torch.log(pi)*chosen_action, dim=-1)
            loss *= advantage[a].squeeze()

            sum_loss += torch.mean(loss).item()
            print("actor loss", sum_loss)

            # compute the gradients of the policy network using the advantage
            # do not use optimizer.zero_grad() since we want to accumulate the gradients for all agents
            loss.backward(torch.ones(self.batch_size, self.seq_len).to(self.device))

        # after computing the gradient for all agents, perform a weight update on the policy network
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()
        # Clear state of actor's GRU Cell
        self.reset_agents()
        return sum_loss / (self.n_agents * self.seq_len)

    def td_one(self):
        """
        updates the critic using td(1)
        :return:
        """

        self.reset_agents()
        # create G with two columns, one for current prediction, second for one step lookahead
        G = torch.zeros((self.batch_size, self.seq_len, 2)).to(self.device)

        # initialize the first column with estimates from the target Q network
        predictions = self.target_critic.forward(self.joint_action_state_pl).squeeze()

        # use detach to assign to numpy array
        G[:, :, 0] = predictions.detach().cpu().numpy()

        sum_loss = 0.
        # by moving backwards, construct the G matrix
        for t in range(self.seq_len - 2, -1, -1):
            G[:, t, 1] = self.reward_seq_pl[:, t] + self.discount * G[:, t + 1, 0]

            # if using soft actor critic, compute the joint action dist for the next time step
            if self.SAC:
                joint_action_dist = 1
                for a in range(self.n_agents):
                    joint_action_dist *= self.agents[a].get_action_dist(self.actor_input_pl[a][:, t + 1, :], eps=0).detach()
                G[:, t, 1] -= self.alpha * torch.sum(torch.log(joint_action_dist) * joint_action_dist, dim=-1)

            pred = self.critic.forward(self.joint_action_state_pl[:, t]).squeeze()

            loss = torch.mean(torch.pow(G[:, t, 1] - pred, 2)) / self.seq_len
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
        self.reset_agents()

        # first compute the future discounted return at every step using the sequence of rewards
        G = np.zeros((self.batch_size, self.seq_len, self.seq_len))

        # initialize the first column with estimates from the target Q network
        # predictions = self.target_critic(self.joint_action_state_pl).squeeze()
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

        sum_loss = 0.0

        # by moving backwards except for the last transition, compute targets and update critic
        for t in range(self.seq_len - 2, -1, -1):

            # vector of powers of lambda
            weights = np.fromfunction(lambda i: lam ** i, shape=(self.seq_len - 1 - t,))

            # normalize
            weights = weights / np.sum(weights)

            # print('t', t)
            targets[:, t] = torch.from_numpy(G[:, t, 1:self.seq_len-t].dot(weights)).to(self.device)

            if self.SAC:
                joint_action_dist = 1
                for a in range(self.n_agents):
                    joint_action_dist *= self.agents[a].get_action_dist(self.actor_input_pl[a][:, t + 1, :], eps=0).detach()
                targets[:, t] -= self.alpha * torch.sum(torch.log(joint_action_dist) * joint_action_dist, dim=-1)

            # print('target', targets[0, t])
            pred = self.critic(self.joint_action_state_pl[:, t]).squeeze()
            # print('pred', pred[0])

            loss = torch.mean(torch.pow(targets[:, t] - pred, 2)) / self.seq_len
            sum_loss += loss.item()
            # print("critic loss", sum_loss)
            # fit the Critic
            self.critic_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.critic_optimizer.step()

        return sum_loss / self.seq_len




