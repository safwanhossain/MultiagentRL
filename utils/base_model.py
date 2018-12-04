from comet_ml import Experiment
import torch
import numpy as np
import multiprocessing

class BaseModel:

    def __init__(self, use_gpu=True, track_results=True):
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
                pi = self.policy(curr_agent_obs[n], actions[n, :], n, eps=0).cpu()
                # sample action from pi, convert to one-hot vector
                action_idx = (torch.multinomial(pi, num_samples=1))
                actions[n, :] = torch.zeros(self.action_size).scatter(0, action_idx, 1)

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
                    pi = self.policy(curr_agent_obs[n], actions[n, :], n, eps).cpu()
                    if torch.sum(pi < 0.) > 0:
                        print("YIKES")
                    # sample action from pi, convert to one-hot vector
                    action_idx = (torch.multinomial(pi, num_samples=1))
                    actions[n, :] = torch.zeros(self.action_size).scatter(0, action_idx, 1)

            count += self.seq_len

        print("Mean reward for this batch: {0:5.3}".format(np.mean(rewards)))
        return np.mean(rewards)

    def train(self):
        """
        Train model
        """
        metrics = {}
        for e in range(self.epochs):
            eps = 0 if self.SAC else max(0.01, 0.15 - 0.15*e/self.epochs)
            metrics["Reward"] = self.gather_batch(eps=eps)
            metrics["Critic Loss"], metrics["Actor Loss"] = self.update(e)
             # self.evaluate()

            self.experiment.log_multiple_metrics(metrics)
            self.experiment.set_step(e)

