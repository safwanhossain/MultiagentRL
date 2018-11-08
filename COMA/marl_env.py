from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import numpy as np

def make_env(scenario_name, n_agents, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
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
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(n_agents)
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    return env

if __name__=="__main__":
    env = make_env('simple_spread', n_agents=1)

    print(env.action_space)
    print(env.observation_space)
    env.reset()


    while True:
        env.render(mode="not human")
        # NOOP RIGHT LEFT UP DOWN, can induce diagonal motion
        obs_n, reward_n, done_n, info_n = env.step([[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        print(reward_n)
        print(len(obs_n[0]))