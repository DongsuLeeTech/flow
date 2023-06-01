
from flow.core import lane_change_rewards as rewards
from flow.core.params import InitialConfig, NetParams
from flow.envs.multiagent.base import MultiEnv

from gym.spaces.box import Box
from gym.spaces.tuple import Tuple
from gym.spaces.multi_discrete import MultiDiscrete

import numpy as np
import random

from collections import defaultdict
from pprint import pprint

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # # lane change duration for autonomous vehicles, in s. Autonomous vehicles
    # # reject new lane changing commands for this duration after successfully
    # # changing lanes.
    # "lane_change_duration": 5,
    # # desired velocity for all vehicles in the network, in m/s
    # "target_velocity": 10,
    # # specifies whether vehicles are to be sorted by position during a
    # # simulation step. If set to True, the environment parameter
    # # self.sorted_ids will return a list of all vehicles sorted in accordance
    # # with the environment
    # 'sort_vehicles': False
    # bounds on the ranges of ring road lengths the autonomous vehicle is
    # trained on
    'ring_length': [220, 270],
}

class TD3NGSIMAccelEnv(MultiEnv):
    """Fully observable lane change and acceleration environment.

    This environment is used to train autonomous vehicles to improve traffic
    flows when lane-change and acceleration actions are permitted by the rl
    agent.

    """
    
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

        lb = [-abs(max_decel), -1]
        ub = [max_accel, 1.]
        shape = self.initial_vehicles.num_rl_vehicles,
        return Box(np.array(lb), np.array(ub), dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=0,
            high=1,
            shape=(3 * self.initial_vehicles.num_vehicles,),
            dtype=np.float32)
            
    def compute_reward(self, rl_actions, **kwargs):

        if rl_actions is None:
            return {}
        
        rls = self.k.vehicle.get_rl_ids()
       
        reward = {}  # reward = 0

        rl_des = self.initial_config.reward_params.get('rl_desired_speed', 0)
        rl_action_p = self.initial_config.reward_params.get('rl_action_penalty', 0)
        uns4IDM_p = self.initial_config.reward_params.get('uns4IDM_penalty', 0)
        mlp = self.initial_config.reward_params.get('meaningless_penalty', 0)

        for rl in rls:
            r = 0
            if rl_des:
                if self.k.vehicle.get_speed(rl) > 0.:
                    vel = np.array([
                        self.k.vehicle.get_speed(rl)
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

                    r = rl_des * (1 - (cost / (max_cost + eps)))
                    # if r > 0:
                    #     r = r ** 2

            reward[rl] = r

        for veh_id in rls:
            pen1 = 0
            if self.k.vehicle.get_leader(veh_id) is not None:
                if mlp:
                    if self.k.vehicle.get_last_lc(veh_id) == self.time_counter:
                        lane_leaders = self.k.vehicle.get_lane_leaders(veh_id)
                        headway = [(self.k.vehicle.get_x_by_id(leader) - self.k.vehicle.get_x_by_id(veh_id))
                                    % self.k.network.length() / self.k.network.length() for leader in lane_leaders]
                        # FOR N LANE
                        if headway[self.k.vehicle.get_previous_lane(veh_id)] \
                                - headway[self.k.vehicle.get_lane(veh_id)] > 0:
                            pen1 -= mlp * (headway[self.k.vehicle.get_previous_lane(veh_id)]
                                           - headway[self.k.vehicle.get_lane(veh_id)]) * 130 \
                                    * (headway[self.k.vehicle.get_previous_lane(veh_id)])

            reward[veh_id] += pen1

        for veh_id in rls:
            pen2 = 0
            if self.k.vehicle.get_follower(veh_id) is not None:
                v = self.k.vehicle.get_speed(veh_id)
                tw = self.k.vehicle.get_tailway(veh_id)
                follow_id = self.k.vehicle.get_follower(veh_id)

                if uns4IDM_p:
                    T = 1
                    a = 1
                    b = 1
                    s0 = 2

                    if abs(tw) < 1e-3:
                        tw = 1e-3

                    if follow_id is None or follow_id == '':
                        s_star = 0

                    else:
                        follow_vel = self.k.vehicle.get_speed(follow_id)
                        s_star = s0 + max(
                            0, follow_vel * T + follow_vel * (follow_vel - v) /
                               (2 * np.sqrt(a * b)))

                    pen2 = uns4IDM_p * max(-5, min(0, 1 - (s_star / tw) ** 2))

                reward[veh_id] = reward[veh_id] + pen2

        for rl in rls:
            if rl_action_p and rl_actions:
                pen3 = 0

                dir = rl_actions[rl][1::2]
                if dir <= -0.333:
                    dir = -1
                elif dir >= 0.333:
                    dir = 1
                else:
                    dir = 0

                if dir:
                    if self.k.vehicle.get_previous_lane(rl) == self.k.vehicle.get_lane(rl):
                        pen3 -= rl_action_p

                reward[rl] = reward[rl] + pen3

        return reward

    def get_state(self):
        """See class definition."""
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
        rl_ids = list(actions.keys())
        acceleration = []
        direction = []
        for i in actions.values():
            acceleration.append(i[::2])
            dir = i[1::2]
            if dir <= -0.333:
                dir = -1
            elif dir >= 0.333:
                dir = 1
            else:
                dir = 0

            direction.append(dir)

        self.k.vehicle.apply_acceleration(rl_ids, acc=acceleration)
        self.k.vehicle.apply_lane_change(rl_ids, direction=direction)

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        if self.k.vehicle.num_rl_vehicles > 0:
            for veh_id in self.k.vehicle.get_human_ids():
                self.k.vehicle.set_observed(veh_id)


class TD3NGSIMAccelPOEnv(TD3NGSIMAccelEnv):
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
            shape=(3 * 2 * 2 + 2, ),
            dtype=np.float32)

    def get_state(self):
        """See class definition."""
        # normalizers
        max_speed = 10 # self.k.network.max_speed() * 2
        length = self.k.network.length()
        max_lanes = max(
            self.k.network.num_lanes(edge)
            for edge in self.k.network.get_edge_list())

        # print(max_speed)

        """See class definition."""
        obs = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            lane_leaders = self.k.vehicle.get_lane_leaders(rl_id) or rl_id
            lane_followers = self.k.vehicle.get_lane_followers(rl_id)

            # Velocity of vehicles
            lane_followers_speed = self.k.vehicle.get_lane_followers_speed(rl_id)
            lane_leaders_speed = self.k.vehicle.get_lane_leaders_speed(rl_id)

            rl_speed = self.k.vehicle.get_speed(rl_id)
            # for i in rl_speed:
            if rl_speed / max_speed > 1:
                rl_speed = 1.

            # Position of Vehicles
            lane_followers_pos = [self.k.vehicle.get_x_by_id(follower) for follower in lane_followers]
            lane_leaders_pos = [self.k.vehicle.get_x_by_id(leader) for leader in lane_leaders]

            for i in range(0, max_lanes):
                if self.k.vehicle.get_lane(rl_id) == i:
                    lane_followers_speed = lane_followers_speed[max(0, i - 1):i + 2]
                    lane_leaders_speed = lane_leaders_speed[max(0, i - 1):i + 2]
                    lane_leaders_pos = lane_leaders_pos[max(0, i - 1):i + 2]
                    lane_followers_pos = lane_followers_pos[max(0, i - 1):i + 2]

                    if i == 0:
                        f_sp = [(rl_speed - speed) / max_speed
                                for speed in lane_followers_speed]
                        f_sp.insert(0, -1.)
                        l_sp = [(rl_speed - speed) / max_speed
                                for speed in lane_leaders_speed]
                        l_sp.insert(0, -1.)
                        f_pos = [((self.k.vehicle.get_x_by_id(rl_id) - pos) % length / length)
                                 for pos in lane_followers_pos]
                        f_pos.insert(0, -1.)
                        l_pos = [(pos - self.k.vehicle.get_x_by_id(rl_id)) % length / length
                                 for pos in lane_leaders_pos]
                        l_pos.insert(0, -1.)
                        lanes = [0.]

                    elif i == max_lanes - 1:
                        f_sp = [(rl_speed - speed) / max_speed
                                for speed in lane_followers_speed]
                        f_sp.insert(2, -1.)
                        l_sp = [(rl_speed - speed) / max_speed
                                for speed in lane_leaders_speed]
                        l_sp.insert(2, -1.)
                        f_pos = [((self.k.vehicle.get_x_by_id(rl_id) - pos) % length / length)
                                 for pos in
                                 lane_followers_pos]
                        f_pos.insert(2, -1.)
                        l_pos = [(pos - self.k.vehicle.get_x_by_id(rl_id)) % length / length
                                 for pos in
                                 lane_leaders_pos]
                        l_pos.insert(2, -1.)
                        lanes = [1.]

                    else:
                        f_sp = [(rl_speed - speed) / max_speed
                                for speed in lane_followers_speed]
                        l_sp = [(rl_speed - speed) / max_speed
                                for speed in lane_leaders_speed]
                        f_pos = [((self.k.vehicle.get_x_by_id(rl_id) - pos) % length / length)
                                 for pos in lane_followers_pos]
                        l_pos = [(pos - self.k.vehicle.get_x_by_id(rl_id)) % length / length
                                 for pos in lane_leaders_pos]
                        lanes = [0.5]

                    rl_sp = [rl_speed / max_speed]
                    positions = l_pos + f_pos
                    speeds = rl_sp + l_sp + f_sp

                    # if none value
                    for i in range(len(speeds)):
                        if speeds[i] > 1.0 or speeds[i] < -1.0:
                            speeds[i] = -1.0
                            positions[i-1] = -1.0

                        else:
                            pass

                    observation = np.array(speeds + positions + lanes)
                    obs.update({rl_id: observation})

        return obs

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        for veh_id in self.visible:
            self.k.vehicle.set_observed(veh_id)