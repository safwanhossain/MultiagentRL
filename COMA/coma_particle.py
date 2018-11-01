"""
Train agents for particle environment using COMA
"""
from coma import COMA
import marl_env
import torch
import numpy as np

def gather_rollouts(coma, eps):
    """
    gathers rollouts under the current policy
    :param coma: instance of COMA model
    :param eps:
    :return:
    """
    # Step 1: generate batch_size rollouts
    for i in range(coma.batch_size):
        done = False
        joint_action = torch.zeros((coma.n_agents, coma.action_size))
        coma.env.reset()

        for t in range(coma.seq_len):

            # get observations
            obs_n, reward_n, done_n, info_n = coma.env.step(joint_action)

            # they all get the same reward, save the reward
            coma.reward_seq_pl[i, t] = reward_n[0]

            # save the joint action
            coma.joint_action_pl[i, t, :] = joint_action.flatten()

            # reset the joint action, one-hot representation
            joint_action.fill_(0)

            # for each agent, save observation, compute next action
            for n in range(coma.n_agents):
                # use the observation to construct global state
                # the global state consists of positions + velocity of agents, first 4 elements from obs
                coma.global_state_pl[i, t, n * 4:4 * n + 4] = torch.from_numpy(obs_n[n][0:4])

                # get distribution over actions
                obs_action = np.concatenate((obs_n[n][0:coma.obs_size], joint_action[n, :]))
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

    # concatenate the joint action, global state, set network inputs to torch tensors
    coma.joint_action_state_pl = torch.cat((coma.joint_action_pl, coma.global_state_pl), dim=-1)

    coma.joint_action_state_pl.requires_grad_(True)

if __name__ == "__main__":

    env = marl_env.make_env('simple_spread')

    n_agents = 3
    n_landmarks = 3

    coma = COMA(env=env, batch_size=1, seq_len=13, discount=0.8, n_agents=3, action_size=5, obs_size=14,
                     state_size=18, h_size=16)


    for e in range(20):

        gather_rollouts(coma, eps=0.05 - e*0.0025)
        coma.fit_critic(lam=0.5)
        coma.fit_actor(eps=0.05)


