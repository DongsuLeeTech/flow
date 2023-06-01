from flow.envs.ring.accel import AccelEnv
from flow.envs.ring.lane_change_accel import LaneChangeAccelEnv
from flow.core import lane_change_rewards as rewards

from gym.spaces.box import Box
from gym.spaces.tuple import Tuple
from gym.spaces.multi_discrete import MultiDiscrete

import numpy as np

from collections import defaultdict
from pprint import pprint

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # lane change duration for autonomous vehicles, in s. Autonomous vehicles
    # reject new lane changing commands for this duration after successfully
    # changing lanes.
    "lane_change_duration": 5,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 10,
    # specifies whether vehicles are to be sorted by position during a
    # simulation step. If set to True, the environment parameter
    # self.sorted_ids will return a list of all vehicles sorted in accordance
    # with the environment
    'sort_vehicles': False
}


class NGSIMAccelEnv(AccelEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        """See class definition."""
        max_decel = self.env_params.additional_params["max_decel"]
        max_accel = self.env_params.additional_params["max_accel"]

        lb = [-abs(max_decel)] * self.initial_vehicles.num_rl_vehicles
        ub = [max_accel] * self.initial_vehicles.num_rl_vehicles
        shape = self.initial_vehicles.num_rl_vehicles,
        return Box(np.array(lb), np.array(ub), dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=0,
            high=1,
            shape=(5 * self.initial_vehicles.num_vehicles,),
            dtype=np.float32)

    def compute_reward(self, actions, **kwargs):
        rls = self.k.vehicle.get_rl_ids()
        rwd = 0

        if actions is None:
            return 0

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in rls
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.

        target_vel = self.env_params.additional_params['target_velocity']
        max_cost = np.array([target_vel])
        max_cost = np.linalg.norm(max_cost)

        cost = vel - target_vel
        cost = np.linalg.norm(cost)

        # epsilon term (to deal with ZeroDivisionError exceptions)
        eps = np.finfo(np.float32).eps

        # reward average velocity
        rwd = (1 - (cost / (max_cost + eps)))

        return rwd

    def get_state(self):
        # normalizers
        max_speed = self.k.network.max_speed()
        length = self.k.network.length()
        max_lanes = max(
            self.k.network.num_lanes(edge)
            for edge in self.k.network.get_edge_list())

        speed = [self.k.vehicle.get_speed(veh_id) / max_speed
                 for veh_id in self.sorted_ids]
        pos = [self.k.vehicle.get_x_by_id(veh_id) / length
               for veh_id in self.sorted_ids]
        lane = [self.k.vehicle.get_lane(veh_id) / max_lanes
                for veh_id in self.sorted_ids]

        return np.array(speed + pos + lane)

    def _apply_rl_actions(self, actions):
        self.k.vehicle.apply_acceleration(
            self.k.vehicle.get_rl_ids(), actions)

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        if self.k.vehicle.num_rl_vehicles > 0:
            for veh_id in self.k.vehicle.get_human_ids():
                self.k.vehicle.set_observed(veh_id)


class NGSIMAccelPOEnv(NGSIMAccelEnv):
    """POMDP version of LaneChangeAccelEnv.

        Required from env_params:

        * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
        * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
        * lane_change_duration: lane change duration for autonomous vehicles, in s
        * target_velocity: desired velocity for all vehicles in the network, in m/s
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        self.num_lanes = max(self.k.network.num_lanes(edge)
                             for edge in self.k.network.get_edge_list())
        self.visible = []

    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=-1,
            high=1,
            shape=(4, ),
            dtype=np.float32)

    def get_state(self):
        """See class definition."""
        # normalizers
        max_speed = 10  # self.k.network.max_speed()
        length = self.k.network.length()
        max_lanes = max(
            self.k.network.num_lanes(edge)
            for edge in self.k.network.get_edge_list())

        # NOTE: this works for only single agent environment
        rl = self.k.vehicle.get_rl_ids()[0]
        lane_followers = self.k.vehicle.get_lane_followers(rl)
        lane_leaders = self.k.vehicle.get_lane_leaders(rl)

        # Velocity of vehicles
        lane_followers_speed = self.k.vehicle.get_lane_followers_speed(rl)
        lane_leaders_speed = self.k.vehicle.get_lane_leaders_speed(rl)
        rl_speed = self.k.vehicle.get_speed(rl)
        # for i in rl_speed:
        if rl_speed / max_speed > 1:
            rl_speed = 1.

        # Position of Vehicles
        lane_followers_pos = [self.k.vehicle.get_x_by_id(follower) for follower in lane_followers]
        lane_leaders_pos = [self.k.vehicle.get_x_by_id(leader) for leader in lane_leaders]

        # Own velocity
        rl_sp = [rl_speed / max_speed]

        # Relative velocity with Heading Vehicle and Following Vehicle
        RVHDWY = [(rl_speed -
                   lane_leaders_speed[self.k.vehicle.get_lane(rl)]) / max_speed]
        RVFOWY = [(rl_speed -
                   lane_followers_speed[self.k.vehicle.get_lane(rl)]) / max_speed]

        # Relative distance with Heading Vehicle and Following Vehicle
        RDHDWY = [(lane_leaders_pos[self.k.vehicle.get_lane(rl)] -
                   self.k.vehicle.get_x_by_id(rl))
                  % length / length]
        # RDFOWY = [(self.k.vehicle.get_x_by_id(rl) -
        #            lane_followers_pos[self.k.vehicle.get_lane(rl)])
        #           % length / length]

        observation = np.array(rl_sp + RVHDWY + RVFOWY + RDHDWY)

        return observation

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        for veh_id in self.visible:
            self.k.vehicle.set_observed(veh_id)

class NGSIMOnlyVelAccelPOEnv(NGSIMAccelEnv):
    """POMDP version of LaneChangeAccelEnv.

        Required from env_params:

        * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
        * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
        * lane_change_duration: lane change duration for autonomous vehicles, in s
        * target_velocity: desired velocity for all vehicles in the network, in m/s
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        self.num_lanes = max(self.k.network.num_lanes(edge)
                             for edge in self.k.network.get_edge_list())
        self.visible = []

    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=-1,
            high=1,
            shape=(1, ),
            dtype=np.float32)

    def get_state(self):
        """See class definition."""
        # normalizers
        max_speed = 10 # self.k.network.max_speed()
        length = self.k.network.length()
        max_lanes = max(
            self.k.network.num_lanes(edge)
            for edge in self.k.network.get_edge_list())

        # NOTE: this works for only single agent environment
        rl = self.k.vehicle.get_rl_ids()[0]
        rl_speed = self.k.vehicle.get_speed(rl)

        # for i in rl_speed:
        if rl_speed / max_speed > 1:
            rl_speed = 1.

        # Own velocity
        rl_sp = [rl_speed / max_speed]

        observation = np.array(rl_sp)
        # print(observation)
        return observation

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        for veh_id in self.visible:
            self.k.vehicle.set_observed(veh_id)
