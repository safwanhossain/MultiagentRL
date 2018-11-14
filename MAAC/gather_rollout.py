"""
Train agents for particle environment using COMA
"""
from train_maac import MAAC
import marl_env
import torch
import numpy as np

def gather_rollouts(maac, num_envs, eps):
    """
    gathers rollouts under the current policy for a given episode
    :param maac: instance of COMA model
    :param num_envs: number of parallel environments
    :param eps:
    :return:
    """
    # Step 1: initialize actions to NOOP
    joint_actions = torch.zeros(len(maac.parallel_envs), maac.n_agents, maac.action_size)
    prev_global_obs_n = []
    reward_arr = torch.zeros(len(maac.parallel_envs), seq_len)

    for l in range(len(maax.parallel_envs)):
        joint_actions[l, :, 0] = 1
    for env in maac.parallel_envs:
        prev_global_obs_n.append(env.reset())
        
    
    # Step 2: Start going thru a sequence of actions under current policy for all parallel environments
    sequence_length = maac.seq_len
    for i in range(sequence_length):
        obs_for_actor = torch.zeros(self.n_agents, len(maac.parallel_envs), maac.obs_size)
        
        for e, env in enumerate(maac.parallel_envs):
            # get observations, by executing current joint action
            obs_n, reward_n, done_n, info_n = env.step(joint_actions[e])
            prev_obs_n = prev_global_obs_n[e]

            # they all get the same reward, save the reward
            reward = reward_n[0]
            reward_arr[e,i] = reward

            # save the joint action for training
            joint_actions = joint_actions[e].flatten()
             
            # Add this obs_n, action, reward, next_obs to the buffer
            maac.add_to_buffer(prev_obs_n, joint_actions, reward, obs_n)

            # store the observations needed for actor forward pass
            for n in range(maac.n_agents):
                obs_for_actor[n,e,:] = torch.from_numpy(obs_n[n][0:maac.obs_size])  

            # current observation becomes previous ones
            prev_global_obs_n[e] = obs_n

        # Step 3: compute the next action
        joint_actions = torch.zeros(len(maac.parallel_envs), maac.n_agents, maac.action_size)
        for n in range(maac.n_agents):
            obs = obs_for_actor[n]
            dist = maac.actor.action(obs)
            
            # sample action from pi, convert to one-hot vector
            action_idx = (torch.multinomial(dist, num_samples=1)).numpy().flatten()
            for l in len(maac.parallel_envs):
                joint_actions[l][n][action_idx[l]] = 1

    # print the reward at the last timestep
    print('reward', torch.mean(reward_arr))

if __name__ == "__main__":

    env = marl_env.make_env('simple_spread')

    n_agents = 3
    n_landmarks = 3

    coma = COMA(env=env, batch_size=20, seq_len=20, discount=0.8, n_agents=3, action_size=5, obs_size=14,
                     state_size=18, h_size=128)


    for e in range(2000):
        print('e', e)

        if e % 10 == 0:
            coma.update_target()
            visualize(coma)

        gather_rollouts(coma, eps=0.05 - e*0.00025)

        print('gathered rollouts')
        coma.fit_critic(lam=0.5)
        coma.fit_actor(eps=0.05 - e*0.00025)




