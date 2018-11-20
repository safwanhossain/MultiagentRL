#!/usr/bin/python3
from utils.base_model import BaseModel
from actors import GRUActor, MLPActor
from environment.particle_env import make_env, visualize
from critic import Critic
import numpy as np

from utils.buffer import Buffer

import torch
import time

""" Run the COMA training regime. Training is done in batch mode with a batch size of 30. Given the 
small nature of the current problem, we are NOT parameter sharing between agents (will do so when 
things start working). In each episode (similar to "epoch" in normal ML parlance), we:
    (1) collect sequences of data (batch size/n); 
    (2) use this batch of samples to compute the Value function; 
    (3) use the Value function and compute the baseline and update the actor and critic weights
    
"""

class COMA(BaseModel):

    def __init__(self, env, critic_arch, policy_arch, batch_size, seq_len, discount, lam,
                 lr_critic=0.0005, lr_actor=0.0001, use_gpu=True):
        """
        Initialize all aspects of the model
        :param env: Environment model will be used in
        :param batch_size:
        :param seq_len: horizon
        :param discount: discounting factor
        :param lam: for TD(lambda) return
        :param h_size: size of GRU state
        """
        super(COMA, self).__init__(use_gpu=use_gpu)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.discount = discount
        self.lam = lam
        self.n_agents = env.n
        self.action_size = env.action_size
        self.obs_size = env.agent_obs_size
        self.state_size = env.global_state_size
        self.env = env
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.epochs = 1500
        self.num_updates = 1
        self.num_entries_per_update = self.batch_size * self.seq_len

        self.critic_arch = critic_arch
        self.policy_arch = policy_arch

        # The buffer to hold all the information of an episode
        self.buffer = Buffer(self.num_entries_per_update, self.seq_len, self.num_entries_per_update,
                             self.n_agents, env.agent_obs_size, env.global_state_size, env.action_size)

        # Create "placeholders" for incoming training data (Sorry, tensorflow habit)
        # joint-action state pairs
        self.joint_action_state_pl = torch.zeros((batch_size, seq_len, self.state_size+self.action_size*self.n_agents))
        self.joint_action_state_pl = self.joint_action_state_pl.to(self.device)

        # obs, prev_action pairs, one tensor for each agent
        self.actor_input_pl = \
            [torch.zeros((batch_size, seq_len, self.obs_size+self.action_size+self.n_agents)).to(self.device)
            for _ in range(self.n_agents)]

        # sequence of immediate rewards
        self.reward_seq_pl = np.zeros((batch_size, seq_len))

        # set up the modules for actor-critic based on specified arch
        # specify GRU or MLP policy
        ActorType = policy_arch['type']
        # actor takes in agent index as well
        self.actor = ActorType(input_size=self.obs_size + self.action_size + self.n_agents,
                               h_size=policy_arch['h_size'],
                               action_size = self.action_size,
                               device=self.device,
                               n_agents = self.n_agents)
        self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=self.lr_actor, eps=1e-08)

        self.critic = Critic(input_size=self.action_size*(self.n_agents) + self.state_size,
                             hidden_size=critic_arch['h_size'],
                             device=self.device,
                             n_layers=critic_arch['n_layers'])
        self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr=self.lr_critic, eps=1e-08)

        self.target_critic = Critic(input_size=self.action_size*(self.n_agents) + self.state_size,
                                    hidden_size=critic_arch['h_size'],
                                    n_layers=critic_arch['n_layers'],
                                    device=self.device)
        self.params["lam"] = self.lam
        self.set_params()
        self.experiment.log_multiple_params(self.policy_arch)
        self.experiment.log_multiple_params(self.critic_arch)

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
        # Clear state of all actor's GRU Cell
        self.actor.reset()
        # first get the Q value for the joint actions
        # print('q input', self.joint_action_state_pl[0, :, :])

        q_vals = self.critic.forward(self.joint_action_state_pl)

        # print("q_vals", q_vals)

        diff_actions = torch.zeros_like(self.joint_action_state_pl).to(self.device)

        # initialize the advantage for each agent, by copying the Q values
        advantage = [q_vals.clone().detach() for a in range(self.n_agents)]

        # Optimizer
        self.actor_optimizer.zero_grad()

        sum_loss = 0.0

        # computing baselines, by broadcasting across rollouts and time-steps
        for a in range(self.n_agents):
            # get the dist over actions, based on each agent's observation, use the same epsilon during fitting?
            action_dist = self.actor(self.actor_input_pl[a], a, eps=0)

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

            # print('a', a, 'action_dist', (action_dist[0, 0, :] + EPS))
            # print('action_dist', action_dist)
            # print('advantage', torch.mean(advantage[a]))

            sum_loss += torch.mean(loss).item()

            # compute the gradients of the policy network using the advantage
            # do not use optimizer.zero_grad() since we want to accumulate the gradients for all agents
            loss.backward(torch.ones(self.batch_size, self.seq_len).to(self.device))

        # after computing the gradient for all agents, perform a weight update on the policy network
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()
        # Clear state of actor's GRU Cell
        self.actor.reset()
        return sum_loss / (self.n_agents * self.seq_len)

    def update_critic(self):
        """
        Updates the critic using the off-line lambda return algorithm
        :return:
        """
        lam = self.lam

        # first compute the future discounted return at every step using the sequence of rewards
        G = np.zeros((self.batch_size, self.seq_len, self.seq_len + 1))

        # initialize the first column with estimates from the target Q network
        predictions = self.target_critic(self.joint_action_state_pl).squeeze()

        # use detach to assign to numpy array
        G[:, :, 0] = predictions.detach().cpu().numpy()

        # by moving backwards, construct the G matrix
        for t in range(self.seq_len - 1, -1, -1):

            # loop from one-step lookahead to pure MC estimate
            for n in range(1, self.seq_len + 1):

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

        for t in range(self.seq_len - 2, -1, -1):

            # vector of powers of lambda
            weights = np.fromfunction(lambda i: lam ** i, shape=(self.seq_len - t,))

            # normalize
            weights = weights / np.sum(weights)

            # print('t', t)
            targets[:, t] = torch.from_numpy(G[:, t, 1:self.seq_len-t+1].dot(weights)).to(self.device)
            # print('target', targets[0, t])
            pred = self.critic(self.joint_action_state_pl[:, t]).squeeze()
            # print('pred', pred[0])

            loss = torch.mean(torch.pow(targets[:, t] - pred, 2)) / self.seq_len
            sum_loss += loss.item()
            # print("critic loss", sum_loss)
            # fit the Critic
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

        return sum_loss / self.seq_len

    def format_buffer_data(self):
        """
        Reshape buffer data to correct dimensions for maac
        """
        self.reward_seq_pl[:, :] = self.buffer.rewards[:, :].numpy()

        self.joint_action_state_pl[:, :, :self.state_size] = self.buffer.curr_global_state[:, :, :]
        self.joint_action_state_pl[:, :, self.state_size:] = self.buffer.actions.view(self.batch_size, self.seq_len, -1)
        self.joint_action_state_pl.requires_grad_(True)

        noops = torch.zeros(self.batch_size, 1, self.n_agents, env.action_size)
        noops[:, 0, :, 0] = 1
        prev_action = torch.cat((noops, self.buffer.actions[:, :-1, :, :]), dim=1)

        for n in range(self.n_agents):
            agent_idx = torch.zeros(self.batch_size, self.seq_len, self.n_agents)
            agent_idx = agent_idx.scatter(2, torch.zeros(agent_idx.shape).fill_(n).long(), 1)

            actor_input = torch.cat((self.buffer.curr_agent_obs[:, :, n, :], prev_action[:, :, n, :], agent_idx), dim=2)
            actor_input = actor_input.view(self.batch_size, self.seq_len, -1).type(torch.FloatTensor)
            self.actor_input_pl[n][:, :, :] = actor_input
        return None

    def policy(self, obs, prev_action, n, eps):
        """
        Return probability distribution over actions given observation
        :param obs: observation from environment for agent n
        :param prev_action: last action of this agent
        :param n: agent index
        :return: probability distribution of type torch tensor
        """
        agent_idx = torch.zeros(self.n_agents).scatter(0, torch.zeros(self.n_agents).fill_(n).long() , 1)
        actor_input = torch.cat((obs, prev_action, agent_idx))
        actor_input = actor_input.view(1, 1, -1).type(torch.FloatTensor).to(self.device)
        return self.actor(actor_input, n, eps=eps)[0][0]

    def update(self, epoch):
        """
        update model
        """
        self.format_buffer_data()

        critic_loss = self.update_critic()
        actor_loss = self.update_actor()
        if epoch % 5 == 0:
            # print('e', epoch)
            self.update_target_network()

        self.buffer.reset()
        return critic_loss, actor_loss


if __name__ == "__main__":
    env = make_env(n_agents=1)

    policy_arch = {'type': MLPActor, 'h_size': 128}
    critic_arch = {'h_size': 128, 'n_layers':2}

    model = COMA(env=env, critic_arch=critic_arch, policy_arch=policy_arch,
                batch_size=30, seq_len=100, discount=0.8, lam=0.8, lr_critic=0.0002, lr_actor=0.0001, use_gpu=True)

    st = time.time()
    model.train()
    print("Time taken for {0:d} epochs {1:10.4f}".format(model.epochs, time.time() - st))

    visualize(model)