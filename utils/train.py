import sys
sys.path.append('../')
from comet_ml import Experiment
import argparse
from environment.particle_env import make_env, visualize
from environment.sc2_env_wrapper import SC2EnvWrapper
import numpy as np
import time
from main_model.coma import COMA
from main_model.maac import MAAC
from main_model.actors import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='True',
                        help='use GPU if True, CPU if False [default: True]')
    parser.add_argument('--maac', default='False',
                        help='Whether to use maac critic or not [default: True]')
    parser.add_argument('--SAC', default='True',
                        help='Whether to use SAC or not [default: True]')
    parser.add_argument('--TD_LAMBDA', default='True',
                        help='Whether to use TD_LAMBDA or TD_ONE [default: True]')
    parser.add_argument('--track_results', default='True',
                        help='Whether to track results on comet or not [default: True]')
    parser.add_argument('--save_model', default='False',
                        help='Whether to save the trained model or not [default: True]')
    parser.add_argument('--num_agents', type=int, default=5,
                        help='Number of agents in particle environment [default: 3]')
    parser.add_argument('--env', default="sc2",
                        help='Environment to run ("sc2" or "particle" [default: particle]')
    parser.add_argument('--num_env', default="1",
                        help='Number of parallel environments (default to 1)')
    parser.add_argument('--maac_advantage', default="False",
                        help='Whether to use the MAAC advantage')
    parser.add_argument('--to_log_csv', default="True",
                        help='Whether to use log resutls to csv files')

    flags = parser.parse_args()

    envs = []
    if flags.env == "particle":
        for i in range(int(flags.num_env)):
            env = make_env(n_agents=flags.num_agents)
            np.random.seed(i*1000)
            env.seed(i*1000)
            envs.append(env)
    elif flags.env == "sc2":
        envs = [SC2EnvWrapper("CollectMineralShards") for _ in range(int(flags.num_env))]
    else:
        raise TypeError("Requested environment does not exist or is not implemented yet")

    policy_arch = {'type': MLPActor, 'h_size': 128}
    critic_arch = {'h_size': 128, 'n_layers':2}

    # Files to log the stats - allows us to compare various models
    reward_file = "reward_sc2_mineral_shards_coma.csv"
    critic_loss_file = "critic_loss.csv"
    agent_loss_file = "agent_loss.csv"

    if flags.maac.lower() in ["true", "t", "yes", "y"]:
        Model = MAAC
    else:
        Model = COMA

    model = Model(flags, envs=envs, critic_arch=critic_arch, policy_arch=policy_arch,
                  batch_size=30, seq_len=80, discount=0.9, lam=0.8, alpha=0.1, lr_critic=0.0002,
                  lr_actor=0.0001, log_files=[reward_file, critic_loss_file, agent_loss_file])

    st = time.time()

    try:
        model.train()
    except KeyboardInterrupt:
        pass

    print("Time taken for {0:d} epochs {1:10.4f}".format(model.epochs, time.time() - st))
    #visualize(model)
