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
        self.experiment = Experiment(api_key='1jl4lQOnJsVdZR6oekS6WO5FI', project_name=self.__class__.__name__,
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
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "discount": self.discount,
            "n_agents": self.n_agents,
            "lambda": self.lam,
            "lr_critic": self.lr_critic,
            "lr_actor": self.lr_actor,
            "alpha": self.alpha,
        })

        self.experiment.log_multiple_params(self.params)

    def gather_batch(self, eps):
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
            self.env.reset()
            self.reset_agents()

            for t in range(self.seq_len):

                # get observations, by executing current joint action
                obs_n, reward_n, done_n, info_n = self.env.step(joint_action)

                # they all get the same reward, save the reward, divided by number of agents, to prevent large rewards and to compare with single agent experiments
                self.reward_seq_pl[i, t] = reward_n[0] / self.n_agents

                # for each agent, save observation, compute next action
                for n in range(self.n_agents):
                    # one-hot agent index
                    agent_idx = np.zeros(self.n_agents)
                    agent_idx[n] = 1

                    # use the observation to construct global state
                    # the global state consists of positions + velocity of agents, first 4 elements from obs
                    self.global_state_pl[i, t, n * 4:4 * n + 4] = torch.from_numpy(obs_n[n][0:4])

                    # get distribution over actions, concatenate observation and prev action for actor training
                    obs_action = np.concatenate((obs_n[n][0:self.obs_size], joint_action[n, :], agent_idx))
                    actor_input = torch.from_numpy(obs_action).type(torch.FloatTensor)

                    # save the actor input for training
                    self.actor_input_pl[n][i, t, :] = actor_input

                    action = self.agents[n].get_action(self.actor_input_pl[n][i, t, :].view(1, 1, -1), eps=eps)
                    joint_action[n, :] = action

                    # save the next joint action for training
                    self.joint_action_pl[i, t, :] = joint_action.flatten()

                # get the absolute landmark positions for the global state
                self.global_state_pl[i, t, self.n_agents * 4:] = torch.from_numpy(np.array(
                    [landmark.state.p_pos for landmark in self.env.world.landmarks]).flatten())

        # concatenate the joint action, global state, set network inputs to torch tensors
        # action taken at state s
        self.joint_action_state_pl = torch.cat((self.joint_action_pl, self.global_state_pl), dim=-1).to(self.device)

        self.joint_action_state_pl.requires_grad_(True)

        # return the mean reward of the batch
        return np.mean(self.reward_seq_pl)


    def train(self, epochs):
        """
        Train model
        """
        if self.SAC:
            eps = 0

        else:
            eps = 0.15

        metrics = {}
        for e in range(epochs):
            metrics["Reward"] = self.gather_batch(eps=eps - e*eps/epochs)
            metrics["Critic Loss"], metrics["Actor Loss"] = self.update(e)

            self.experiment.log_multiple_metrics(metrics)
            self.experiment.set_step(e)

