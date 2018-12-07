from comet_ml import Experiment
import torch
import numpy as np
import multiprocessing

class BaseModel:

    def __init__(self, use_gpu=True, track_results=False):
        """
        Initializes comet tracking and cuda gpu
        """
        super().__init__()
        self.experiment = Experiment(api_key='1jl4lQOnJsVdZR6oekS6WO5FI', project_name='sc2',#self.__class__.__name__,
                                     auto_param_logging=False, auto_metric_logging=False,
                                     disabled=(not track_results))
        self.use_gpu = use_gpu

        self.device = torch.device('cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.params = {}

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
            "n_agents": self.env.n,
            "lr_critic": self.lr_critic,
            "lr_actor": self.lr_actor,
            "MAAC": self.use_maac,
            "SAC": self.SAC
        })
        self.experiment.log_multiple_params(self.params)

    def evaluate(self):
        rewards = []
        true_rewards = []

        # initialize action to noop
        actions = torch.zeros(self.n_agents, self.env.action_size)
        actions[:, 0] = 1
        # np.random.seed()
        curr_agent_obs, curr_global_state, reward, true_reward, _ = self.env.reset()

        for t in range(self.seq_len):
            # get observations, by executing current action
            # TODO add parallelism
            next_agent_obs, next_global_state, reward, true_reward, _ = self.env.step(actions)
            rewards.append(reward)
            true_rewards.append(true_reward)
            for n in range(self.n_agents):
                pi = self.policy(curr_agent_obs[n], actions[n, :], n, eps=0).cpu()
                # sample action from pi, convert to one-hot vector
                action_idx = (torch.multinomial(pi, num_samples=1))
                actions[n, :] = torch.zeros(self.action_size).scatter(0, action_idx, 1)

        print("Mean reward for this batch: {0:5.3}".format(np.sum(true_rewards) / float(self.batch_size)))
        return (np.sum(rewards) / float(self.batch_size), np.sum(true_rewards) / float(self.batch_size))

    def gather_batch(self, eps):
        """
        Fills data buffer with (o, o', s, s', a, r) tuples generated from simulating environment
        """
        rewards = []
        true_rewards = []
        for b in range(self.batch_size):
            # initialize action to noop
            actions = torch.zeros(self.n_agents, self.env.action_size)
            actions[:, 0] = 1
            # np.random.seed()
            curr_agent_obs, curr_global_state, reward, true_reward, _ = self.env.reset()

            for t in range(self.seq_len):
                # get observations, by executing current action
                # TODO add parallelism
                next_agent_obs, next_global_state, reward, true_reward, end_signal = self.env.step(actions)

                self.buffer.add_to_buffer(b, t, curr_agent_obs, next_agent_obs, curr_global_state, next_global_state,
                                           actions, reward)
                curr_agent_obs, curr_global_state = next_agent_obs, next_global_state
                rewards.append(reward)
                true_rewards.append(true_reward)

                if end_signal:
                    # self.buffer.set_end_index(b, t)
                    print("HIT RESET", t)
                    break

                # for each agent, save observation, compute next action
                for n in range(self.n_agents):
                    pi = self.policy(curr_agent_obs[n], actions[n, :], n, eps).cpu()

                    # sample action from pi, convert to one-hot vector
                    action_idx = (torch.multinomial(pi, num_samples=1))
                    actions[n, :] = torch.zeros(self.action_size).scatter(0, action_idx, 1)
            self.buffer.set_end_index(b, t)

        print("Mean reward for this batch: {0:5.3}".format(np.mean(rewards)))
        return (np.sum(rewards) / float(self.batch_size), np.sum(true_rewards) / float(self.batch_size))

    def train(self):
        """
        Train model
        """
        metrics = {}
        for e in range(self.epochs):
            eps = 0. if self.SAC else max(0.01, 0.15 - 0.15*e/self.epochs)
            metrics["Reward"], metrics["True Reward"] = self.gather_batch(eps=eps)
            metrics["Critic Loss"], metrics["Actor Loss"] = self.update(e)
             # self.evaluate()

            self.experiment.log_multiple_metrics(metrics, step=e+1)

