import numpy as np

from pysc2.lib import features
from pysc2.env import sc2_env

from s2clientprotocol import raw_pb2 as raw_pb
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import common_pb2 as common_pb

import torch

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

NUM_MOVE_ACTIONS = 4

class SC2EnvWrapper:

    mini_games = {
        "DefeatRoaches": {
            "NUM_ALLIES": 9,  # Used to determine input size to NN
            "NUM_OTHERS": 4,  # Used to determine input size to NN
            "NUM_TOTAL": 13,
            "ACTION_SIZE": NUM_MOVE_ACTIONS + 4, # 2 +  #2 for noop and stop, NUM_MOVE_ACTIONS for movements, 4 to target enemies,
            "X_MAP_SIZE": 63,  # Map size is 64x64, with indices between 0 and 63
            "Y_MAP_SIZE": 63,
            "COMBAT": True
        },

        "CollectMineralShards": {
            "NUM_ALLIES": 2,  # Used to determine input size to NN
            "NUM_OTHERS": 20,  # Used to determine input size to NN
            "NUM_TOTAL": 22,
            "ACTION_SIZE": NUM_MOVE_ACTIONS, # 2 + , # 2 for noop and stop, NUM_MOVE_ACTIONS for movement
            "X_MAP_SIZE": 63, # Map size is 64x64, with indices between 0 and 63
            "Y_MAP_SIZE": 63,
            "COMBAT": False
        }
    }

    unit_type_mapping = {
        48: 1,
        110: 2,
        1680: 3
    }

    def __init__(self, map, step_mul=5, visualize=True):
        """
        :param map[string]: map to use
        :param agent_type[agent object type]: agent to use
        :param max_timesteps[int]: maximum number of timesteps to run per episode
        :param max_episodes[int]: number of episodes to run
        :param step_mul: How many game steps per agent step (action/observation). None means use the map default.
        :param visualize: Whether to visualize the episodes
        """
        self.aif = sc2_env.AgentInterfaceFormat(feature_dimensions=sc2_env.Dimensions(screen=64, minimap=24),
                                                use_raw_units=True,
                                                camera_width_world_units=24)
        self.map = map
        self.step_mul = step_mul
        self.visualize = visualize

        self.mg_info = SC2EnvWrapper.mini_games[map]

        self.n = self.mg_info["NUM_ALLIES"]
        self.action_size = self.mg_info["ACTION_SIZE"]
        self.obs_size = 6 if self.mg_info["COMBAT"] else 4
        self.agent_obs_size = (self.mg_info["NUM_TOTAL"] - 1) * self.obs_size
        self.global_state_size = self.mg_info["NUM_TOTAL"] * (self.obs_size - 1)
        self.env = sc2_env.SC2Env(map_name=self.map,
                   agent_interface_format=self.aif,
                   step_mul=self.step_mul,
                   visualize=self.visualize)
        self.agent_obs = None
        self.global_obs = None
        self.unit_tags = np.zeros(self.mg_info["NUM_TOTAL"], dtype=np.int64)
        self.num_crystals = self.mg_info["NUM_TOTAL"]

    def reset(self):
        """
        Creates an agent per unit to control and provides requisite inputs
        :param env: starcraft environment
        :return: return the environments reset timestep
        """
        ts = self.env.reset()
        self.num_crystals = self.mg_info["NUM_TOTAL"]
        return self.transform_env_info(ts)

    def transform_env_info(self, timestep):
        """
        Returns individual agent observations and global observations for the critic
        NUM_TOTAL = NUM_ALLIES + NUM_OTHERS
        Global observations will be numpy arrays of size [NUM_TOTAL * 5]
        Agent observations will be numpy arrays of size [NUM_TOTAL * 6]
        The first NUM_ALLIES entries will describe allied units (within each units field of vision for agent obs)
        The next NUM_OTHER entries will describe neutral/enemy (based on minigame) units
        (within the units field of vision for agent obs)
        Each agent entry will have 6 fields: unit_type, relative x, relative y, distance, health, shield
        Global entries will be identical except lacking a distance

        :param timestep: full observation from game
        :return: individual agent observations and global observations for the critic
        """
        obs = timestep[0].observation
        agent_observations = np.zeros([self.mg_info["NUM_ALLIES"], self.agent_obs_size], dtype=np.float32)
        global_observations = np.zeros([self.mg_info["NUM_TOTAL"], self.obs_size - 1], dtype=np.float32)
        # Get xy of all units
        raw_units = sorted(obs.raw_units, key=lambda u: u.tag)
        xy = np.array([[unit.x, unit.y] for unit in raw_units])

        global_idx_self = 0
        global_idx_other = self.mg_info["NUM_ALLIES"]
        for i, unit in enumerate(raw_units):
            if unit.alliance == _PLAYER_SELF:
                gi = global_idx_self
                global_idx_self += 1
            else:
                gi = global_idx_other
                global_idx_other += 1

            ut = SC2EnvWrapper.unit_type_mapping.get(unit.unit_type)
            if ut is None:
                raise TypeError("unit type mapping not found. Create unit mapping for unit type", unit.unit_type)

            if self.mg_info["COMBAT"]:
                global_observations[gi] = [ut,
                                           unit.x,
                                           unit.y,
                                           unit.health,
                                           unit.shield]
            else:
                global_observations[gi] = [ut,
                                           unit.x,
                                           unit.y]
            self.unit_tags[gi] = unit.tag

            if unit.alliance != _PLAYER_SELF:
                continue

            agent_obs = np.zeros([self.mg_info["NUM_TOTAL"] - 1, self.obs_size])
            # precalculate all distances usig matrices
            xy_distances = [unit.x, unit.y] - xy
            distances = np.sqrt(np.sum(np.square(xy_distances), axis=1))
            # setup indexing
            obs_self_idx = 0
            obs_enemy_idx = self.mg_info["NUM_ALLIES"] - 1

            for unit_idx, other_unit in enumerate(raw_units):
                # Only consider units within vision
                if distances[unit_idx] > 9: # taken from https://liquipedia.net/starcraft2/Sight for marine sight
                    continue
                # Ignore own unit
                if unit is other_unit:
                    continue
                # Determine where to put data
                if other_unit.alliance != _PLAYER_SELF:
                    obs_idx = obs_enemy_idx
                    obs_enemy_idx += 1
                else:
                    obs_idx = obs_self_idx
                    obs_self_idx += 1

                ut = SC2EnvWrapper.unit_type_mapping.get(other_unit.unit_type)
                if ut is None:
                    print("unit type mapping not found. Create unit mapping for unit type", other_unit.unit_type)
                    raise TypeError

                if self.mg_info["COMBAT"]:
                    agent_obs[obs_idx] = [ut,
                                          xy_distances[unit_idx][0],
                                          xy_distances[unit_idx][1],
                                          distances[unit_idx],
                                          other_unit.health,
                                          other_unit.shield]
                else:
                    agent_obs[obs_idx] = [ut,
                                          xy_distances[unit_idx][0],
                                          xy_distances[unit_idx][1],
                                          distances[unit_idx]]

            agent_observations[global_idx_self - 1] = agent_obs.flatten()

        self.global_obs, self.agent_obs = global_observations, agent_observations

        reward = timestep[0].reward
        num_crystals = global_idx_other - 1
        if num_crystals > self.num_crystals:
            print("CRYSTALS CLEARED")
            reward += 10
            self.num_crystals = num_crystals


        return torch.from_numpy(agent_observations), \
               torch.from_numpy(global_observations.flatten()), \
               torch.from_numpy(np.array(reward, dtype=np.float32)),\
               timestep[0].last()

    def step(self, action_indices):
        """
        Run the environment
        :param action_indices: pytorch tensor shape=(num_agents, num_actions) with one hot encoding of actions per agent
        :return: agent observations, global observations, reward
        """
        # Change one hot encoding actions into sc2 actions
        actions = []
        for n in range(self.n):
            index = action_indices[n].nonzero()[0][0].item()
            unit_tag = self.unit_tags[n]
            xy = (self.global_obs[n][1], self.global_obs[n][2])
            unit_cmd = None
            # if index == 0:
            #     unit_cmd = self.noop(unit_tag)
            # elif index == 1:
            #     unit_cmd = self.stop(unit_tag)
            if 0 <= index < NUM_MOVE_ACTIONS:
                unit_cmd = self.move(unit_tag, index, xy)
            elif index > NUM_MOVE_ACTIONS + 1:
                unit_cmd = self.attack(unit_tag, index, xy)

            raw_action = raw_pb.ActionRaw(unit_command=unit_cmd)
            actions.append(sc_pb.Action(action_raw=raw_action))

        # run sc2 environment with actions
        timestep = self.env.step([actions])
        # get interpretable observations and rewards
        return self.transform_env_info(timestep)

    def move(self, unit_tag, index, xy):
        """
        create starcraft move action
        :param unit_tag: unit tag of agent
        :param index: action index
        :param xy: current xy location of unit
        :return: move action
        """
        index = index
        angle = 2. * np.pi * index / NUM_MOVE_ACTIONS
        x = int(round(xy[0] + np.cos(angle) * 100))
        y = int(round(xy[1] + np.sin(angle) * 100))

        x = np.clip(x, 0, self.mg_info["X_MAP_SIZE"])
        y = np.clip(y, 0, self.mg_info["Y_MAP_SIZE"])

        point = common_pb.Point2D(x=x, y=y)
        return raw_pb.ActionRawUnitCommand(ability_id = 16,
                                           unit_tags = [unit_tag],
                                           queue_command = False,
                                           target_world_space_pos = point)

    def attack(self, unit_tag, index, xy):
        """
        create starcraft attack action
        :param unit_tag: unit tag of agent
        :param index: action index
        :param xy: current xy location of unit
        :return: attack action
        """
        target_idx = self.mg_info["NUM_ALLIES"] + (index - NUM_MOVE_ACTIONS - 1)
        target_xy = (self.global_obs[target_idx][1], self.global_obs[target_idx][2])
        # Out of Marine range see https://liquipedia.net/starcraft2/Marine_(Legacy_of_the_Void)
        if np.sqrt(np.sum(np.square(np.subtract(xy, target_xy)))) > 5:
            return None
        target_tag = self.unit_tags[target_idx]
        return raw_pb.ActionRawUnitCommand(ability_id = 23,
                                           unit_tags = [unit_tag],
                                           queue_command = False,
                                           target_unit_tag = target_tag)

    def stop(self, unit_tag):
        """
        create starcraft stop action
        :return: stop action
        """
        return raw_pb.ActionRawUnitCommand(ability_id = 4,
                                           unit_tags=[unit_tag],
                                           queue_command = False)

    def noop(self, unit_tag):
        return None

    def __del__(self):
        self.env.close()

