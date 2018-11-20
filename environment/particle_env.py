from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import numpy as np
import torch
import time

def visualize(model):
    actions = torch.zeros((model.n_agents, model.action_size))
    actions[:, 0] = 1
    model.env.reset()

    for t in range(model.seq_len):
        model.env.render()
        time.sleep(0.05)
        # get observations
        obs_n, state, reward = model.env.step(actions)

        # reset the joint action, one-hot representation
        actions = torch.zeros((model.n_agents, model.action_size))

        # for each agent, save observation, compute next action
        for n in range(model.n_agents):

            pi = model.policy(obs_n[n], actions[n, :], n, eps=0).cpu()

            # sample action from pi, convert to one-hot vector
            action_idx = torch.multinomial(pi, num_samples=1)
            actions[n, :] = torch.zeros(model.action_size).scatter(0, action_idx, 1)

class ParticleEnv(MultiAgentEnv):
    """
    Particle environment specifically for multi agent simple spread environment using pytorch.
    Number of landmarks will always equal the number of agents
    Action space:
        NOOP
        LEFT
        RIGHT
        UP
        DOWN
        Total 5

    Agent Observation space:
        Own velocity (x, y)
        Own position (x, y)
        Landmark positions with respect to self n * (x, y)
        The positions of other agents with respect self (n - 1) * (x, y)
        Total = (2 + 2 + 2n + 2(n - 1)) = 4n + 2

    Global State space:
        Position of each agent n * (x, y)
        Velocity of each agent n * (x, y)
        Landmark positions n * (x, y)
        Total = 3n * 2 = 6n
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_size = 5
        self.agent_obs_size = 4 * self.n + 2
        self.global_state_size = 6 * self.n

    def transform_env_info(self, obs_n, reward_n):
        agent_obs = np.array(obs_n, dtype=np.float32)[:, 0:self.agent_obs_size]
        global_state = np.zeros(self.global_state_size, dtype=np.float32)
        global_state[:4 * self.n] = agent_obs[:, 0:4].flatten()
        global_state[4 * self.n:] = self.landmarks[:]
        reward = reward_n[0] / self.n
        return torch.from_numpy(agent_obs), torch.from_numpy(global_state), torch.from_numpy(np.array(reward, dtype=np.float32))

    def step(self, *args, **kwargs):
        obs_n, reward_n, done_n, info_n = super().step(*args, **kwargs)
        return self.transform_env_info(obs_n, reward_n)

    def reset(self, *args, **kwargs):
        obs_n = super().reset(*args, **kwargs)
        self.landmarks = np.array([landmark.state.p_pos for landmark in self.world.landmarks]).flatten()
        return self.transform_env_info(obs_n, np.zeros((self.n)))


def make_env(n_agents, benchmark=False):
    '''
    Creates a ParticleEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.
    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)
    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    # load scenario from script
    scenario = scenarios.load("simple_spread.py").Scenario()
    # create world
    world = scenario.make_world(n_agents)
    # create multiagent environment
    if benchmark:
        env = ParticleEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = ParticleEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    return env

if __name__=="__main__":
    env = make_env(n_agents=1)

    print(env.action_space)
    print(env.observation_space)
    env.reset()


    while True:
        env.render(mode="not human")
        # NOOP RIGHT LEFT UP DOWN, can induce diagonal motion
        obs_n, reward_n, done_n, info_n = env.step([[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        print(reward_n)
        print(len(obs_n[0]))