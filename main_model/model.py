#!/usr/bin/python3
import argparse
from utils.base_model import BaseModel
from actors import GRUActor, MLPActor
from environment.particle_env import make_env, visualize
from environment.sc2_env_wrapper import SC2EnvWrapper
from critic_maac import Critic as MAAC_Critic
from critic import Critic
import numpy as np

from utils.buffer import Buffer

import torch
import time

""" Run the main_model training regime. Training is done in batch mode with a batch size of 30. Given the 
small nature of the current problem, we are NOT parameter sharing between agents (will do so when 
things start working). In each episode (similar to "epoch" in normal ML parlance), we:
    (1) collect sequences of data (batch size/n); 
    (2) use this batch of samples to compute the Value function; 
    (3) use the Value function and compute the baseline and update the actor and critic weights
    
"""

class Model(BaseModel):

    def __init__(self, flags, env, critic_arch, policy_arch, batch_size, seq_len, discount, lam,
                 lr_critic=0.0001, lr_actor=0.0001):
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
        super(Model, self).__init__(use_gpu=use_gpu, track_results=track_results)
        self.use_maac = flags.maac.lower() in ["true", "t", "yes", "y"]
        self.SAC = flags.SAC.lower() in ["true", "t", "yes", "y"]
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.discount = discount
        self.lam = lam
        self.n_agents = env.n
        self.action_size = env.action_size
        self.obs_size = env.agent_obs_size
        self.state_size = env.global_state_size
        self.alpha = 0.1
        self.env = env
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.epochs = 10000
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

        self.observations = torch.zeros(self.n_agents, self.batch_size, self.seq_len, self.obs_size).to(self.device)
        self.actions = torch.zeros(self.n_agents, self.batch_size, self.seq_len, self.action_size).to(self.device)

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

        if self.use_maac:
            self.critic = MAAC_Critic(observation_size=self.obs_size, action_size=self.action_size,
                                      num_agents=self.n_agents, attention_heads=4, embedding_dim=128, device=self.device)

            self.target_critic = MAAC_Critic(observation_size=self.obs_size, action_size=self.action_size,
                                      num_agents=self.n_agents, attention_heads=4, embedding_dim=128, device=self.device)

        else:
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

        self.params["lam"] = self.lam
        self.set_params()
        self.experiment.log_multiple_params(self.policy_arch)
        self.experiment.log_multiple_params(self.critic_arch)

    def get_critic_input(self, t=None):
        if t is None:
            return (self.observations, self.actions) if self.use_maac else self.joint_action_state_pl
        else:
            return (self.observations[:, :, t:t + 1, :], self.actions[:, :, t:t + 1, :]) if self.use_maac else \
                   self.joint_action_state_pl[:, t]


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

        q_vals = self.critic.forward(self.get_critic_input()).detach()

        # print("q_vals", q_vals)
        diff_action_in = self.actions if self.use_maac else self.joint_action_state_pl
        diff_actions = torch.zeros_like(diff_action_in).to(self.device)

        # initialize the advantage for each agent, by copying the Q values
        if self.use_maac:
            advantage = [q_vals[a].clone().detach() for a in range(self.n_agents)]
        else:
            advantage = [q_vals.clone().detach() for a in range(self.n_agents)]


        # Optimizer
        self.actor_optimizer.zero_grad()

        sum_loss = 0.0
        EPS = 1.0e-08

        # computing baselines, by broadcasting across rollouts and time-steps
        for a in range(self.n_agents):
            # get the dist over actions, based on each agent's observation, use the same epsilon during fitting?
            pi = self.actor(self.actor_input_pl[a], a, eps=0)

            # make a copy of the joint-action, state, to substitute different actions in it
            diff_actions.copy_(diff_action_in)

            # get the chosen action for each agent
            action_index = a * self.action_size
            chosen_action = self.joint_action_state_pl[:, :, action_index:action_index+self.action_size]

            # compute the baseline for that agent, by substituting different actions in the joint action
            for u in range(self.action_size):
                action = torch.zeros(self.action_size)
                action[u] = 1

                # index into that agent's action to substitute a different one
                if self.use_maac:
                    diff_actions[a, :, :, :] = action
                    critic_in = (self.observations, diff_actions)
                else:
                    diff_actions[:, :, action_index:action_index+self.action_size] = action
                    critic_in = diff_actions


                # get the Q value of that new joint action
                # Q_u = self.critic.forward(self.joint_action_state_pl)
                Q_u = self.critic(critic_in)

                if self.use_maac:
                    Q_u = Q_u[a]

                advantage[a] -= Q_u*pi[:, :, u].unsqueeze_(-1)

            # elif self.use_maac:
            #     maac_baseline = (all_q[a].to(self.device) * pi.to(self.device)).sum(dim=-1, keepdim=True)
            #     advantage[a] = q_vals[a] - maac_baseline
                # advantage[a] = (advantage[a] - advantage[a].mean()) / advantage[a].std()  # make it 0 mean and 1 var (idk why??)

            if self.SAC:
                entropy = torch.sum(torch.log(pi), dim=-1).unsqueeze_(-1)
                advantage[a] -= self.alpha * entropy

            # loss is negative log of probability of chosen action, scaled by the advantage
            # the advantage is treated as a scalar, so the gradient is computed only for log prob
            advantage[a] = advantage[a].detach()

            # sum along the action dim, left with log loss for chosen action
            loss = -torch.sum(torch.log(pi)*chosen_action, dim=-1)
            loss *= advantage[a].squeeze()

            sum_loss += torch.mean(loss).item()

            # compute the gradients of the policy network using the advantage
            # do not use optimizer.zero_grad() since we want to accumulate the gradients for all agents
            loss.backward(torch.ones(self.batch_size, self.seq_len).to(self.device))

        # after computing the gradient for all agents, perform a weight update on the policy network
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()
        # Clear state of actor's GRU Cell
        self.actor.reset()
        return sum_loss / (self.n_agents)

    def td_one(self):
        """
        updates the critic using td(1)
        :return:
        """

        # create G with two columns, one for current prediction, second for one step lookahead
        G = np.zeros((self.n_agents, self.batch_size, self.seq_len, 2))

        # initialize the first column with estimates from the target Q network
        predictions = self.target_critic.forward(self.get_critic_input()).squeeze()

        # use detach to assign to numpy array
        G[:, :, :, 0] = predictions.detach().cpu().numpy()

        sum_loss = 0.
        # by moving backwards, construct the G matrix
        for t in range(self.seq_len - 1, -1, -1):
            G[:, :, t, 1] = self.reward_seq_pl[:, t] + self.discount * G[:, :, t + 1, 0]

            # if using soft actor critic, compute the joint action dist for the next time step
            if self.SAC:
                joint_action_dist = 1
                for a in range(self.n_agents):
                    joint_action_dist *= self.actor(self.actor_input_pl[a][:, t + 1, :], eps=0)
                    G[:, :, t, 1] -= self.alpha * torch.sum(torch.log(joint_action_dist), dim=-1)

            pred = self.critic.forward(self.get_critic_input(t)).squeeze()

            loss = torch.mean(torch.pow(G[:, :, t, 1] - pred, 2)) / self.seq_len
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
        G = np.zeros((self.n_agents, self.batch_size, self.seq_len, self.seq_len))

        # initialize the first column with estimates from the target Q network
        # predictions = self.target_critic(self.joint_action_state_pl).squeeze()
        predictions = self.target_critic.forward(self.get_critic_input()).squeeze()

        # use detach to assign to numpy array
        G[:, :, :, 0] = predictions.detach().cpu().numpy()

        # by moving backwards, construct the G matrix
        for t in range(self.seq_len - 1, -1, -1):

            # loop from one-step lookahead to pure MC estimate
            for n in range(1, self.seq_len - 1):

                # pure MC
                if t + n > self.seq_len - 1:
                    G[:, :, t, n] = self.reward_seq_pl[:, t:].dot(np.fromfunction(lambda i: self.discount**i,
                                                                                 shape=(self.seq_len-t,)))

                # combination of MC + bootstrapping
                else:
                    G[:, :, t, n] = self.reward_seq_pl[:, t] + self.discount*G[:, :, t+1, n-1]

        # compute target at timestep t
        targets = torch.zeros((self.n_agents, self.batch_size, self.seq_len), dtype=torch.float32).to(self.device)

        sum_loss = 0.0

        # by moving backwards except for the last transition, compute targets and update critic
        for t in range(self.seq_len - 2, -1, -1):

            # vector of powers of lambda
            weights = np.fromfunction(lambda i: lam ** i, shape=(self.seq_len - 1 - t,))

            # normalize
            weights = weights / np.sum(weights)

            # print('t', t)
            targets[:, :, t] = torch.from_numpy(G[:, :, t, 1:self.seq_len-t].dot(weights)).to(self.device)

            if self.SAC:
                joint_action_dist = 1
                for a in range(self.n_agents):
                    joint_action_dist *= self.actor(self.actor_input_pl[a][:, t + 1, :], n, eps=0)
                targets[:, :, t] -= self.alpha * torch.sum(torch.log(joint_action_dist) * joint_action_dist, dim=-1)

            # print('target', targets[0, t])
            pred = self.critic(self.get_critic_input(t)).squeeze()
            # print('pred', pred[0])

            loss = torch.mean(torch.pow(targets[:, :, t] - pred, 2))
            sum_loss += loss.item()
            # print("critic loss", sum_loss)
            # fit the Critic
            self.critic_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.critic_optimizer.step()

        return sum_loss / self.seq_len

    def format_buffer_data(self):
        """
        Reshape buffer data to correct dimensions for maac
        """
        self.reward_seq_pl[:, :] = self.buffer.rewards[:, :].numpy()

        self.joint_action_state_pl[:, :, :self.action_size * self.n_agents] = self.buffer.actions.view(self.batch_size, self.seq_len, -1)
        self.joint_action_state_pl[:, :, self.action_size * self.n_agents:] = self.buffer.next_global_state[:, :, :]

        self.observations[:, :, :, :] = self.buffer.next_agent_obs.permute(2, 0, 1, 3)
        self.actions[:, :, :, :] = self.buffer.actions.permute(2, 0, 1, 3)

        for n in range(self.n_agents):
            agent_idx = torch.zeros(self.batch_size, self.seq_len, self.n_agents)
            agent_idx = agent_idx.scatter(2, torch.zeros(agent_idx.shape).fill_(n).long(), 1)

            actor_input = torch.cat((self.buffer.next_agent_obs[:, :, n, :], self.buffer.actions[:, :, n, :], agent_idx), dim=2)
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

        if epoch % 2 == 0:
            # print('e', epoch)
            self.update_target_network()

        critic_loss = self.td_lambda()
        actor_loss = self.update_actor()

        self.buffer.reset()
        return critic_loss, actor_loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='True',
                        help='use GPU if True, CPU if False [default: True]')
    parser.add_argument('--maac', default='False',
                        help='Whether to use maac critic or not [default: True]')
    parser.add_argument('--SAC', default='True',
                        help='Whether to use SAC or not [default: True]')
    parser.add_argument('--track_results', default='True',
                        help='Whether to track results on comet or not [default: True]')
    parser.add_argument('--num_agents', type=int, default=3,
                        help='Number of agents in particle environment [default: 3]')
    parser.add_argument('--env', default="sc2",
                        help='Environment to run ("sc2" or "particle" [default: particle]')

    flags = parser.parse_args()


    if flags.env == "particle":
        env = make_env(n_agents=flags.num_agents)
    elif flags.env == "sc2":
        env = SC2EnvWrapper("CollectMineralShards")
    else:
        raise TypeError("Requested environment does not exist or is not implemented yet")

    policy_arch = {'type': MLPActor, 'h_size': 256}
    critic_arch = {'h_size': 256, 'n_layers': 3}

    model = Model(flags, env=env, critic_arch=critic_arch, policy_arch=policy_arch,
                  batch_size=20, seq_len=400, discount=0.95, lam=0.8, lr_critic=0.00001, lr_actor=0.0001)

    st = time.time()

    try:
        model.train()
    except KeyboardInterrupt:
        pass

    print("Time taken for {0:d} epochs {1:10.4f}".format(model.epochs, time.time() - st))
    visualize(model)