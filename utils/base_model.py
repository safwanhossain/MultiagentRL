from comet_ml import Experiment
import torch
import numpy as np
import multiprocessing
import csv

class BaseModel:

    def __init__(self, use_gpu=True, track_results=True, log_files=None):
        """
        Initializes comet tracking and cuda gpu
        """
        super().__init__()
        self.experiment = Experiment(api_key='1jl4lQOnJsVdZR6oekS6WO5FI', \
                project_name='all_in_one_safwan',#self.__class__.__name__,
                                     auto_param_logging=False, auto_metric_logging=False,
                                     disabled=(not track_results))
        self.use_gpu = use_gpu
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.params = {}
        
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
        })
        self.experiment.log_multiple_params(self.params)

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
                action = self.policy(curr_agent_obs[n], actions[n, :], n, eps=0).cpu()
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
                    action = self.policy(curr_agent_obs[n], actions[n, :], n, eps).cpu()
                    actions[n, :] = action

            count += self.seq_len

        print("Mean reward for this batch: {0:5.3}".format(np.mean(rewards)))
        return np.mean(rewards)

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

    def train(self):
        """
        Train model
        """
        metrics = {}
        for e in range(self.epochs):
            eps = 0 if self.SAC else max(0.01, 0.15 - 0.15*e/self.epochs)
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

