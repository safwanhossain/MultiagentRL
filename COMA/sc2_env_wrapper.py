from environment import Environment
from test_agent import TestAgent

import numpy as np

from pysc2.lib import features
from pysc2.env import sc2_env

from s2clientprotocol import raw_pb2 as raw_pb
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import common_pb2 as common_pb

import time

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

class SC2EnvWrapper:

    # Store this in a better spot
    NUM_ALLIES = 10  # Used to determine input size to NN
    NUM_OTHERS = 20  # Used to determine input size to NN
    NUM_TOTAL = 40

    def __init__(self, map, agent_type, max_timesteps, max_episodes, step_mul=None, visualize=False):
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
        self.max_timesteps = max_timesteps
        self.max_episodes = max_episodes
        self.step_mul = step_mul or 1
        self.visualize = visualize
        self.agent_type = agent_type
        self.agents = []


    def setup(self, env):
        """
        Creates an agent per unit to control and provides requisite inputs
        :param env: starcraft environment
        :return: return the environments reset timestep
        """
        ts = env.reset()
        obs = ts[0].observation
        enemy_units = [unit.tag for unit in obs.raw_units if unit.alliance == _PLAYER_ENEMY]
        for unit in obs.raw_units:
            if unit.alliance == _PLAYER_SELF:
                self.agents.append(SCIIAgentWrapper(unit.tag, self.agent_type, enemy_units))
        return ts

    def allot_observations(self, timestep):
        """
        Returns individual agent observations and global observations for the critic
        Agent observations will be numpy arrays of size [NUM_ALLIES + NUM_OTHERS, 6]
        The first NUM_ALLIES entries will describe allied units within each units field of vision
        The next NUM_OTHER entries will describe neutral/enemy (based on minigame) units within the units field of vision
        Each unit entry will have 6 fields: unit_type, relative x, relative y, distance, health, shield

        :param timestep: full observation from game
        :return: individual agent observations and global observations for the critic
        """
        obs = timestep[0].observation
        agent_observations = []
        global_observations = np.zeros([SC2EnvWrapper.NUM_TOTAL, 5])
        # Get xy of all units
        xy = np.zeros([len(obs.raw_units), 2])

        global_idx_self = 0
        global_idx_other = SC2EnvWrapper.NUM_TOTAL // 2
        unit_ids = []
        for i, unit in enumerate(obs.raw_units):

            xy[i] = [unit.x, unit.y]
            unit_ids.append(unit.tag)

            if unit.alliance == _PLAYER_SELF:
                gi = global_idx_self
                global_idx_self += 1
            else:
                gi = global_idx_other
                global_idx_other += 1

            global_observations[gi] = [unit.unit_type,
                                       unit.x,
                                       unit.y,
                                       unit.health,
                                       unit.shield]

        xy = np.array([[unit.x, unit.y] for unit in obs.raw_units])

        for agent in self.agents:
            agent_unit = obs.raw_units[unit_ids.index(agent.id)]
            agent_obs = np.zeros([SC2EnvWrapper.NUM_ALLIES + SC2EnvWrapper.NUM_OTHERS, 6])
            # precalculate all distances usig matrices
            xy_distances = np.square([agent_unit.x, agent_unit.y] - xy)
            distances = np.sqrt(np.sum(xy_distances, axis=1))
            # setup indexing
            obs_self_idx = 0
            obs_enemy_idx = SC2EnvWrapper.NUM_ALLIES

            for unit_idx, unit in enumerate(obs.raw_units):
                # Only consider units within vision
                if distances[unit_idx] > 9: # taken from https://liquipedia.net/starcraft2/Sight for marine sight
                    continue
                # Ignore own unit
                if unit.tag == agent.id:
                    continue
                # Determine where to put data
                unit = obs.raw_units[unit_idx]
                if unit.alliance != _PLAYER_SELF:
                    obs_idx = obs_enemy_idx
                    obs_enemy_idx += 1
                else:
                    obs_idx = obs_self_idx
                    obs_self_idx += 1

                agent_obs[obs_idx] = [unit.unit_type,
                                      xy_distances[unit_idx][0],
                                      xy_distances[unit_idx][1],
                                      distances[unit_idx],
                                      unit.health,
                                      unit.shield]

            agent_observations.append(agent_obs)

        return agent_observations, global_observations


    def start(self):
        """
        Run the environment
        :return: None
        """
        total_frames = 0
        with sc2_env.SC2Env(map_name=self.map,
                            agent_interface_format=self.aif,
                            step_mul=self.step_mul,
                            game_steps_per_episode=self.max_timesteps * self.step_mul//2,
                            visualize=self.visualize) as env:

            start_time = time.time()
            try:
                timesteps = self.setup(env)

                for episode in range(self.max_episodes):
                    ### RUN 1 SIMULATION ###
                    for step in range(self.max_timesteps):
                        total_frames += 1
                        agent_obs, global_obs = self.allot_observations(timesteps)
                        actions = [agent.step(obs) for agent, obs in zip(self.agents, agent_obs)]
                        # loss = critic(global_obs, actions)
                        if timesteps[0].last():
                            break
                        timesteps = env.step([actions])

                    ### RESET ###
                    timesteps = env.reset()
                    for a in self.agents:
                        a.reset()

            except KeyboardInterrupt:
                pass
            finally:
                elapsed_time = time.time() - start_time
                print("Took %.3f seconds for %s steps: %.3f fps" % (
                    elapsed_time, total_frames, total_frames / elapsed_time))

class SCIIAgentWrapper:
    ARG_SLICE = [slice(0, 2), slice(2, 3), slice(0, 0), slice(0, 0)]

    def __init__(self, unit_tag, agent_type, enemy_ids):
        """
        :param unit_tag[int]: unit tag of unit this agent will control
        :param agent_type: What type of agent to initialize
        :param enemy_ids: T
        """
        self.id = unit_tag
        self.agent = agent_type(len(enemy_ids))
        self.actions = [self.move, self.attack, self.stop, self.noop]
        self.enemy_ids = enemy_ids
        self.index = 0

    def reset(self):
        """
        reset agents
        :return: None
        """
        self.agent.reset()

    def step(self, obs):
        """
        :param obs: observation to pass to agent
        :return: returns starcraft action based on observation
        """
        action_id, arguments = self.agent.step(obs)
        args = arguments[SCIIAgentWrapper.ARG_SLICE[action_id]]
        raw_action = raw_pb.ActionRaw(unit_command=self.actions[action_id](*args))
        action = sc_pb.Action(action_raw=raw_action)
        return action

    def move(self, x, y):
        """
        create starcraft move action
        :param x: x location to move to
        :param y: y location to move to
        :return: move action
        """
        assert (0 <= x <= 64 and 0 <= y <= 64)
        point = common_pb.Point2D(x=x, y=y)
        return raw_pb.ActionRawUnitCommand(ability_id = 16,
                                           unit_tags = [self.id],
                                           queue_command = False,
                                           target_world_space_pos = point)

    def attack(self, target_idx):
        """
        create starcraft attack action
        :param target_idx: target_idx to attack
        :return: attack action
        """
        assert 0 <= target_idx <= len(self.enemy_ids)
        return raw_pb.ActionRawUnitCommand(ability_id = 23,
                                           unit_tags = [self.id],
                                           queue_command = False,
                                           target_unit_tag = self.enemy_ids[target_idx])

    def stop(self):
        """
        create starcraft stop action
        :return: stop action
        """
        return raw_pb.ActionRawUnitCommand(ability_id = 4,
                                           unit_tags=[self.id],
                                           queue_command = False)

    def noop(self):
        return None


if __name__ == "__main__":
    sc2 = SC2EnvWrapper("FindAndDefeatZerglings", TestAgent, 1000, 1, visualize=True)
    sc2.start()
