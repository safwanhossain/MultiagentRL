import sys
sys.path.append('../')
from utils.buffer import Buffer
from abc import ABC, abstractmethod
from comet_ml import Experiment
import torch
import numpy as np
import csv
import utils.files
import os

class BaseModel:

    def __init__(self, envs, batch_size, seq_len, discount, lam,
                 lr_critic=0.0001, lr_actor=0.0001, alpha=0.1, use_gpu=True, track_results=True, log_files=None):
        """
        Initializes comet tracking and cuda gpu
        """
        super().__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.discount = discount
        self.lam = lam
        self.n_agents = envs[0].n
        self.action_size = envs[0].action_size
        self.obs_size = envs[0].agent_obs_size
        self.state_size = envs[0].global_state_size
        self.alpha = alpha
        self.envs = envs
        self.num_envs = len(envs)
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.epochs = 5000
        self.num_updates = 1
        self.num_entries_per_update = self.batch_size * self.seq_len

        self.use_gpu = use_gpu
        self.device = torch.device('cuda:1' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.params = {}

        # The buffer to hold all the information of an episode
        self.buffer = Buffer(self.num_entries_per_update, self.seq_len, self.num_entries_per_update,
                             self.n_agents, self.obs_size, self.state_size, self.action_size)

        # Create "placeholders" for incoming training data (Sorry, tensorflow habit)
        # joint-action state pairs
        self.joint_action_state_pl = torch.zeros(
            (batch_size, seq_len, self.state_size + self.action_size * self.n_agents))
        self.joint_action_state_pl = self.joint_action_state_pl.to(self.device)

        # the global state
        self.global_state_pl = torch.zeros((batch_size, seq_len, self.state_size))

        # joint action of all agents, flattened
        self.joint_action_pl = torch.zeros((batch_size, seq_len, self.action_size * self.n_agents))

        # obs, prev_action pairs, one tensor for each agent
        self.actor_input_pl = \
            [torch.zeros((batch_size, seq_len, self.obs_size + self.action_size + self.n_agents)).to(self.device)
             for _ in range(self.n_agents)]

        self.observations = torch.zeros(self.n_agents, self.batch_size, self.seq_len, self.obs_size).to(self.device)
        self.actions = torch.zeros(self.n_agents, self.batch_size, self.seq_len, self.action_size).to(self.device)

        # sequence of immediate rewards
        self.reward_seq_pl = np.zeros((batch_size, seq_len))


        self.experiment = Experiment(api_key='1jl4lQOnJsVdZR6oekS6WO5FI', \
                project_name='all_in_one_elsa',#self.__class__.__name__,
                                     auto_param_logging=False, auto_metric_logging=False,
                                     disabled=(not track_results))
        
        # This is for logging to a csv file. Since logging can be expensive, we will maintain
        # relevant info in a buffer and empty the buffer into the file after a number of episodes
        self.reward_dict = {}
        self.critic_loss_dict = {}
        self.agent_loss_dict = {}
        if log_files != None:
            self.reward_file, self.critic_loss_file, self.agent_loss_file = log_files

    def set_params(self):
        """
        Create parameter dictionary from necessary parameters and logs them to comet.
        Requires that model has initialized these necessary parameters i.e. run this at the end of the init
        """
        self.params.update({
            "num_agents": self.n_agents,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "discount": self.discount,
            "n_agents": self.envs[0].n,
            "lr_critic": self.lr_critic,
            "lr_actor": self.lr_actor,
            "alpha": self.alpha,
            "SAC": self.SAC,
            "TD_LAMBDA": self.TD_LAMBDA,
            "MAAC": self.use_maac,
        })
        self.experiment.log_multiple_params(self.params)

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path + "_actor.pt"))
        self.critic.load_state_dict(torch.load(path + "_critic.pt"))

    def save_model(self, model_name=None):
        if model_name is None:
            model_name = "maac" if self.use_maac else "coma"

        save_path = utils.files.get_models_path()
        torch.save(self.critic.state_dict(), os.path.join(save_path, self.experiment.get_key() + model_name + "_critic.pt"))
        torch.save(self.actor.state_dict(), os.path.join(save_path, self.experiment.get_key() + model_name + "_actor.pt"))

    def policy(self, obs, prev_action, n, eps):
        """
        Return probability distribution over actions given observation
        :param obs: observation from environment for agent n
        :param prev_action: last action of this agent
        :param n: agent index
        :return: probability distribution of type torch tensor
        """
        agent_idx = torch.zeros(self.n_agents).scatter(0, torch.zeros(self.n_agents).fill_(n).long(), 1)
        actor_input = torch.cat((obs, prev_action, agent_idx))
        actor_input = actor_input.view(1, 1, -1).type(torch.FloatTensor).to(self.device)

        return self.actor(actor_input, eps=eps)[0][0]

    def evaluate(self):
        rewards = []

        # initialize action to noop
        actions = torch.zeros(self.n_agents, self.action_size)
        actions[:, 0] = 1
        # np.random.seed()

        # randomly select an env for evaluation
        env = self.envs[np.random.choice(self.num_envs)]
        curr_agent_obs, curr_global_state, reward = env.reset()

        for t in range(self.seq_len):
            # get observations, by executing current action
            # TODO add parallelism
            next_agent_obs, next_global_state, reward = env.step(actions)
            rewards.append(reward)
            for n in range(self.n_agents):
                # get a chosen action directly (as one hot vector), sampled from pi
                agent_idx = torch.zeros(self.n_agents).scatter(0, torch.zeros(self.n_agents).fill_(n).long(), 1)
                actor_input = torch.cat((curr_agent_obs[n], actions[n, :], agent_idx)).view(1, 1, -1)\
                    .type(torch.FloatTensor).to(self.device)

                action = self.agents[n].get_action(actor_input, eps=0).cpu()
                actions[n, :] = action

        print("Mean reward for this batch: {0:5.3}".format(np.mean(rewards)))
        return np.mean(rewards)

    def gather_batch(self, eps):
        """
        Fills data buffer with (o, o', s, s', a, r) tuples generated from simulating environment
        """
        count = 0
        rewards = []
        while count < self.num_entries_per_update:
            # initialize action to noop
            actions = torch.zeros(self.n_agents, self.action_size)
            actions[:, 0] = 1
            # np.random.seed()

            # select a random environment to simulate
            env = self.envs[np.random.choice(self.num_envs)]
            curr_agent_obs, curr_global_state, reward, _ = env.reset()

            for t in range(self.seq_len):
                # get observations, by executing current action
                # TODO add parallelism
                next_agent_obs, next_global_state, reward, end_signal = env.step(actions)

                self.buffer.add_to_buffer(t, curr_agent_obs, next_agent_obs, curr_global_state, next_global_state,
                                           actions, reward)
                curr_agent_obs, curr_global_state = next_agent_obs, next_global_state
                rewards.append(reward)

                if end_signal:
                    break

                # for each agent, save observation, compute next action
                for n in range(self.n_agents):
                    # get a chosen action directly (as one hot vector), sampled from pi
                    agent_idx = torch.zeros(self.n_agents).scatter(0, torch.zeros(self.n_agents).fill_(n).long(), 1)
                    actor_input = torch.cat((curr_agent_obs[n], actions[n, :], agent_idx)).view(1, 1, -1) \
                        .type(torch.FloatTensor).to(self.device)

                    action = self.agents[n].get_action(actor_input, eps=eps).cpu()
                    actions[n, :] = action

            count += self.seq_len

        print("Mean reward for this batch: {0:5.3}".format(np.mean(rewards)))
        return np.mean(rewards)

    def gather_rollout(self, eps):
        """
           gathers rollouts under the current policy
           saves data in format compatible with coma training
           global state computation is specific to simple-spread environment
           TODO: make env agnostic, add helper functions specific to each environment
           :param self: instance of coma model
           :param eps:
           :return:
           """
        for i in range(self.batch_size):

            # initialize action to noop
            joint_action = torch.zeros((self.n_agents, self.action_size))
            joint_action[:, 0] = 1

            env = self.envs[np.random.choice(self.num_envs)]
            env.reset()
            self.reset_agents()

            for t in range(self.seq_len):

                # get observations, by executing current joint action
                obs_n, global_state, reward_n, _ = env.step(joint_action)

                # they all get the same reward, save the reward
                self.reward_seq_pl[i, t] = reward_n[0]

                self.global_state_pl[i, t, :] = global_state

                # for each agent, save observation, compute next action
                for n in range(self.n_agents):
                    # one-hot agent index
                    agent_idx = torch.zeros(self.n_agents)
                    agent_idx[n] = 1

                    # get distribution over actions, concatenate observation and prev action for actor training
                    obs_action = torch.cat((obs_n[n][0:self.obs_size], joint_action[n, :], agent_idx), -1)
                    actor_input = obs_action

                    # save the actor input for training
                    self.actor_input_pl[n][i, t, :] = actor_input

                    action = self.agents[n].get_action(self.actor_input_pl[n][i, t, :].view(1, 1, -1), eps=eps)
                    joint_action[n, :] = action

                    # save the next joint action for training
                    self.joint_action_pl[i, t, :] = joint_action.flatten()

        # concatenate the joint action, global state, set network inputs to torch tensors
        # action taken at state s
        self.joint_action_state_pl = torch.cat((self.joint_action_pl, self.global_state_pl), dim=-1).to(self.device)

        self.joint_action_state_pl.requires_grad_(True)

        # return the mean reward of the batch
        print("Mean reward for this batch: {0:5.3}".format(np.mean(self.reward_seq_pl)))
        return np.mean(self.reward_seq_pl)

    def log_values_to_file(self):
        with open(self.reward_file, 'a', newline='') as reward_csv:
            writer = csv.writer(reward_csv)
            for e, reward in self.reward_dict.items():
                writer.writerow(["Episode", e, "Reward", reward])
        self.reward_dict = {} 
        
        with open(self.critic_loss_file, 'a', newline='') as critic_csv:
            writer = csv.writer(critic_csv)
            for e, c_loss in self.critic_loss_dict.items():
                writer.writerow(["Episode", e, "Critic loss", c_loss])
        self.critic_loss_dict = {} 

        with open(self.agent_loss_file, 'a', newline='') as actor_csv:
            writer = csv.writer(actor_csv)
            for e, a_loss in self.agent_loss_dict.items():
                writer.writerow(["Episode", e, "Actor loss", a_loss])
        self.actor_reward_dict = {}

    @abstractmethod
    def update_target_network(self):
        pass

    @abstractmethod
    def update_critic(self):
        pass

    @abstractmethod
    def update_actor(self):
        pass

    def format_buffer_data(self):
        """
        Reshape buffer data to correct dimensions for maac
        """
        self.reward_seq_pl[:, :] = self.buffer.rewards[:, :].numpy()

        self.joint_action_state_pl[:, :, :self.action_size * self.n_agents] = self.buffer.actions.view(self.batch_size,
                                                                                                       self.seq_len, -1)
        # pair actions with the state at which they were taken
        self.joint_action_state_pl[:, :, self.action_size * self.n_agents:] = self.buffer.curr_global_state[:, :, :]

        # observations are one step ahead of actions here, because they will be paired and fed to the actor network
        self.observations[:, :, :, :] = self.buffer.next_agent_obs.permute(2, 0, 1, 3)
        self.actions[:, :, :, :] = self.buffer.actions.permute(2, 0, 1, 3)

        for n in range(self.n_agents):
            agent_idx = torch.zeros(self.batch_size, self.seq_len, self.n_agents)
            agent_idx = agent_idx.scatter(2, torch.zeros(agent_idx.shape).fill_(n).long(), 1)

            actor_input = torch.cat(
                (self.buffer.next_agent_obs[:, :, n, :], self.buffer.actions[:, :, n, :], agent_idx), dim=2)
            actor_input = actor_input.view(self.batch_size, self.seq_len, -1).type(torch.FloatTensor)
            self.actor_input_pl[n][:, :, :] = actor_input

        return None

    def update(self, epoch):
        """
        update model
        """
        self.format_buffer_data()

        if epoch % 2 == 0:
            # print('e', epoch)
            self.update_target_network()

        critic_loss = self.update_critic()
        actor_loss = self.update_actor()

        self.buffer.reset()
        return critic_loss, actor_loss

    def train(self):
        """
        Train model
        """
        metrics = {}
        for e in range(self.epochs):

            # eps has to be annealed to zero as the agents get better
            eps = 0 if self.SAC else max(0, 0.15 - 0.15*e/self.epochs)
            metrics["Reward"] = self.gather_batch(eps=eps)
            metrics["Critic Loss"], metrics["Actor Loss"] = self.update(e)
            
            self.reward_dict[e] = metrics["Reward"]
            self.critic_loss_dict[e] = metrics["Critic Loss"]
            self.agent_loss_dict[e] = metrics["Actor Loss"]
            
            # self.evaluate()

            self.experiment.log_multiple_metrics(metrics)
            self.experiment.set_step(e)
            
            if (e % 50) == 0:
                self.log_values_to_file()
                self.save_model()
