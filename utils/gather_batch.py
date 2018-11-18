import torch
import numpy as np
import multiprocessing

def gather_batch(env, model):
    """
    Fills data buffer with (o, o', s, s', a, r) tuples generated from simulating environment
    :param env: environment to simulate
    :param model: model providing agent policies who will use the generated data
    :return:
    """
    count = 0
    rewards = []
    while count < model.batch_size:
        # initialize action to noop
        actions = torch.zeros(model.n_agents, env.action_size)
        actions[:, 0] = 1
        np.random.seed()
        curr_agent_obs, curr_global_state, reward = env.reset()

        for t in range(model.seq_len):
            # for each agent, save observation, compute next action
            for n in range(model.n_agents):
                pi = model.policy(curr_agent_obs[n], actions[n, :], n)
                # sample action from pi, convert to one-hot vector
                action_idx = torch.multinomial(pi, num_samples=1).cuda()
                actions[n, :] = torch.zeros(model.action_size).cuda().scatter(0, action_idx, 1)

            # get observations, by executing current joint action
            #TODO add parallelism
            next_agent_obs, next_global_state, reward = env.step(actions)
            model.buffer.add_to_buffer(t, curr_agent_obs, next_agent_obs, curr_global_state, next_global_state,
                                       actions, reward)
            curr_agent_obs, curr_global_state = next_agent_obs, next_global_state
            rewards.append(reward)

        count += model.seq_len
    print("Mean reward for this batch: {0:10.3}".format(np.mean(rewards)))
