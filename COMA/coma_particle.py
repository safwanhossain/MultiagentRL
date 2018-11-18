"""
Train agents for particle environment using COMA
"""
from comet_ml import Experiment
from coma import COMA
from actor import *
import marl_env
import torch
import time
import numpy as np
import matplotlib.pyplot as plt

def gather_rollouts(coma, eps):
    """
    gathers rollouts under the current policy
    :param coma: instance of COMA model
    :param eps:
    :return:
    """
    # Step 1: generate batch_size rollouts
    for i in range(coma.batch_size):

        # initialize action to noop
        joint_action = torch.zeros((coma.n_agents, coma.action_size))
        joint_action[:, 0] = 1
        coma.env.reset()

        actor_reset = True
        for t in range(coma.seq_len):

            # get observations, by executing current joint action
            obs_n, reward_n, done_n, info_n = coma.env.step(joint_action)

            # they all get the same reward, save the reward, divided by number of agents, to prevent large rewards and to compare with single agent experiments
            coma.reward_seq_pl[i, t] = reward_n[0] / coma.n_agents

            # save the joint action for training
            coma.joint_action_pl[i, t, :] = joint_action.flatten()

            # for each agent, save observation, compute next action
            for n in range(coma.n_agents):
                # one-hot agent index
                agent_idx = np.zeros(coma.n_agents)
                agent_idx[n] = 1

                # use the observation to construct global state
                # the global state consists of positions + velocity of agents, first 4 elements from obs
                coma.global_state_pl[i, t, n * 4:4 * n + 4] = torch.from_numpy(obs_n[n][0:4])

                # get distribution over actions, concatenate observation and prev action
                obs_action = np.concatenate((obs_n[n][0:coma.obs_size], joint_action[n, :], agent_idx))
                actor_input = torch.from_numpy(obs_action).view(1, 1, -1).type(torch.FloatTensor)

                # save the actor input for training
                coma.actor_input_pl[n][i, t, :] = actor_input

                pi = coma.actor.forward(actor_input, eps)

                # sample action from pi, convert to one-hot vector
                action_idx = (torch.multinomial(pi[0, 0, :], num_samples=1)).numpy()
                action = torch.zeros(coma.action_size)
                action[action_idx] = 1
                joint_action[n, :] = action

            # get the absolute landmark positions for the global state
            coma.global_state_pl[i, t, coma.n_agents * 4:] = torch.from_numpy(np.array(
                [landmark.state.p_pos for landmark in coma.env.world.landmarks]).flatten())

            actor_reset = False

    # concatenate the joint action, global state, set network inputs to torch tensors
    coma.joint_action_state_pl = torch.cat((coma.joint_action_pl, coma.global_state_pl), dim=-1)

    coma.joint_action_state_pl.requires_grad_(True)
    # print the reward at the last timestep
    # print('reward', np.mean(np.sum(coma.reward_seq_pl, axis=1)))
    coma.metrics['mean_reward'] = np.mean(coma.reward_seq_pl)

def visualize(coma):

    joint_action = torch.zeros((coma.n_agents, coma.action_size))
    joint_action[:, 0] = 1
    coma.env.reset()

    actor_reset = True
    for t in range(coma.seq_len):
        coma.env.render()
        time.sleep(0.05)
        # get observations
        obs_n, reward_n, done_n, info_n = env.step(joint_action)

        # reset the joint action, one-hot representation
        joint_action = np.zeros((coma.n_agents, coma.action_size))

        # for each agent, save observation, compute next action
        for n in range(coma.n_agents):
            # one-hot agent index
            agent_idx = np.zeros(coma.n_agents)
            agent_idx[n] = 1

            # get distribution over actions
            obs_action = np.concatenate((obs_n[n][0:coma.obs_size], joint_action[n, :], agent_idx))
            actor_input = torch.from_numpy(obs_action).view(1, 1, -1).type(torch.FloatTensor)

            pi = coma.actor.forward(actor_input, eps=0)

            # sample action from pi, convert to one-hot vector
            action_idx = (torch.multinomial(pi[0, 0, :], num_samples=1)).numpy()
            action = np.zeros(coma.action_size)
            action[action_idx] = 1
            joint_action[n, :] = action

        actor_reset = False

if __name__ == "__main__":

    n = 1
    obs_size = 4 + 2*(n-1) + 2*n
    state_size = 4*n + 2*n

    env = marl_env.make_env('simple_spread', n_agents=n)

    policy_arch = {'type': MLPActor, 'h_size': 128}
    critic_arch = {'h_size': 128, 'n_layers': 2}

    coma = COMA(env=env, critic_arch=critic_arch, policy_arch=policy_arch,
                batch_size=30, seq_len=100, discount=0.8, lam=0.8, n_agents=n, action_size=5, obs_size=obs_size,
                     state_size=state_size, lr_critic=0.0005, lr_actor=0.0001)
    #
    experiment = Experiment(api_key='1jl4lQOnJsVdZR6oekS6WO5FI', project_name="COMA", \
                                auto_param_logging=False, auto_metric_logging=False,
                                log_graph=False, log_env_details=False, parse_args=False,
                                auto_output_logging=False)

    experiment.log_multiple_params(coma.params)
    experiment.log_multiple_params(coma.policy_arch)
    experiment.log_multiple_params(coma.critic_arch)

    # visualize(coma)
    try:
        for e in range(4000):
            if e % 2 == 0:
                print('e', e)
                coma.update_target()

            gather_rollouts(coma, eps=0.15 - e*0.05/4000)

            critic_loss = coma.fit_critic()
            print("loss", critic_loss)

            coma.fit_actor()

            # print("reward: {0:5.2f}, actor loss: {1:5.2f}, critic loss: {2:5.2f}".format(
            #     coma.metrics['mean_reward'],
            #     coma.metrics['mean_actor_loss'],
            #     coma.metrics['mean_critic_loss']))

            experiment.set_step(e)
            experiment.log_multiple_metrics(coma.metrics)

    except KeyboardInterrupt:
        visualize(coma)
        pass

    visualize(coma)
    # finally:
    #     plt.plot(coma.metrics['mean_reward'], 'b')
    #     plt.plot(coma.metrics['mean_actor_loss'], 'g')
    #     plt.plot(coma.metrics['mean_critic_loss'], 'r')
    #     plt.show()
#     visualize(coma)