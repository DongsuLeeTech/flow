# from flow.envs.multiagent.ring.accel import AccelEnv
# from flow.envs.ring.lane_change_accel import LaneChangeAccelEnv
from flow.core import UnifiedReward as rewards
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
    "max_accel": 3,
    "max_decel": 3,
    'ring_length': [220, 270],
}

class UniRingAccelEnv(MultiEnv):
    coef_log = []
    
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)

        self.length = self.k.network.length()

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
        global r, pen1, pen2, pen3
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
                if self.k.vehicle.get_speed(rl) >= 0.:
                    r = rewards.rl_desired_speed(self, rl)

            reward[rl] = r

            pen1 = 0
            if mlp:
                lane_leaders = self.k.vehicle.get_lane_leaders(rl)
                headway = [(self.k.vehicle.get_x_by_id(leader) - self.k.vehicle.get_x_by_id(rl))
                           % self.k.network.length() / self.k.network.length() for leader in lane_leaders]
                # if headway[self.k.vehicle.get_previous_lane(rl)] > headway[self.k.vehicle.get_lane(rl)]:
                pen1 = mlp * (headway[self.k.vehicle.get_previous_lane(rl)] - headway[self.k.vehicle.get_lane(rl)])
            reward[rl] -= pen1

            pen2 = 0
            if uns4IDM_p:
                follower = self.k.vehicle.get_follower(rl)
                follow_vel = self.k.vehicle.get_speed(follower)
                pen2 = rewards.unsafe_distance_penalty4IDM(self, rl, self.k.vehicle.get_tailway(rl), follow_vel)

            reward[rl] += pen2

            pen3 = 0
            if rl_action_p:
                pen3 = rewards.rl_action_penalty(self, rl_actions, rl)

            reward[rl] += pen3
            # print(r, pen1, pen2, pen3)
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


class UniRingAccelPOEnv(UniRingAccelEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        self.num_lanes = max(self.k.network.num_lanes(edge)
                             for edge in self.k.network.get_edge_list())
        self.visibility_length = self.initial_config.reward_params.get('visibility_length', 0)
        self.vehicle_length = 5
        self.visibility_lane = self.initial_config.reward_params.get('visibility_lane', 0)
        self.visible = []

    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=-1,
            high=1,
            shape=(self.visibility_lane*5+4,),
            dtype=np.float32,)

    def get_state(self):
        """See class definition."""
        # normalizers
        max_speed = self.k.network.max_speed()
        length = self.k.network.length()
        max_lanes = max(
            self.k.network.num_lanes(edge)
            for edge in self.k.network.get_edge_list())

        # #  Execution
        # rls = self.k.vehicle.get_rl_ids()
        # TV = {}
        #
        # for k in range(len(rls)):
        #
        #     TV[rls[k]] = self.initial_config.reward_params.get('target_velocity', 0)[k]

        """See class definition."""
        obs = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            # tv = TV[rl_id]
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

            # All Leader Position list
            all_ids = np.array(self.k.vehicle.get_ids())
            all_ids = all_ids[all_ids != rl_id]
            all_pos = np.array(self.k.vehicle.get_x_by_id(all_ids))

            if self.k.vehicle.get_x_by_id(rl_id) < self.visibility_length:
                all_pos[all_pos > length - self.visibility_length] -= length

            elif self.k.vehicle.get_x_by_id(rl_id) > length - self.visibility_length:
                all_pos[all_pos < self.visibility_length] += length

            relative_pos = all_pos - self.k.vehicle.get_x_by_id(rl_id)

            lf_ids = all_ids[(relative_pos >= 0) & (relative_pos <= self.visibility_length)]
            lf_lanes = np.array(self.k.vehicle.get_lane(lf_ids))
            lf_dict = dict()
            for lf in range(max_lanes):
                lf_dict[lf] = {'id': []}
            for i in range(len(lf_lanes)):
                lf_dict[lf_lanes[i]]['id'].append(lf_ids[i])

            # Maximum the number of vehicles
            max_density = self.visibility_length // (self.vehicle_length + 5)

            # The number of Lane
            if self.k.vehicle.get_x_by_id(rl_id) > length - self.visibility_length:
                visible_leader_edge = self.k.network.get_edge(self.k.vehicle.get_x_by_id(rl_id)
                                                              + self.visibility_length - length)[0]
            else:
                visible_leader_edge = self.k.network.get_edge(self.k.vehicle.get_x_by_id(rl_id)
                                                              + self.visibility_length)[0]

            cur_num_lane = [self.k.network.num_lanes(self.k.vehicle.get_edge(rl_id)) / max_lanes]
            next_num_lane = [self.k.network.num_lanes(visible_leader_edge) / max_lanes]

            for i in range(0, max_lanes):
                # print(max_lanes)
                if self.k.vehicle.get_lane(rl_id) == i:
                    lane_followers_speed = lane_followers_speed[max(0, i - 1):i + (self.visibility_lane - 1)]
                    lane_leaders_speed = lane_leaders_speed[max(0, i - 1):i + (self.visibility_lane - 1)]
                    lane_leaders_pos = lane_leaders_pos[max(0, i - 1):i + (self.visibility_lane - 1)]
                    lane_followers_pos = lane_followers_pos[max(0, i - 1):i + (self.visibility_lane - 1)]
                    lane_leader_density = [len(lf_dict[i]['id'])/max_density for i in range(0, max_lanes)]

                    if i == 0:
                        f_sp = [(speed - rl_speed) / max_speed
                                for speed in lane_followers_speed]
                        f_sp.insert(0, -1.)
                        l_sp = [(speed - rl_speed) / max_speed
                                for speed in lane_leaders_speed]
                        l_sp.insert(0, +1.)
                        f_pos = [-((self.k.vehicle.get_x_by_id(rl_id) - pos) % length / self.visibility_length)
                                 for pos in lane_followers_pos]
                        f_pos.insert(0, -1.)
                        l_pos = [(pos - self.k.vehicle.get_x_by_id(rl_id)) % length / self.visibility_length
                                 for pos in lane_leaders_pos]
                        l_pos.insert(0, +1.)
                        lanes = [0.]

                    elif i == max_lanes - 1:
                        f_sp = [(speed - rl_speed) / max_speed
                                for speed in lane_followers_speed]
                        f_sp.insert(max_lanes - 1, -1.)
                        l_sp = [(speed - rl_speed) / max_speed
                                for speed in lane_leaders_speed]
                        l_sp.insert(max_lanes - 1, +1.)
                        f_pos = [-((self.k.vehicle.get_x_by_id(rl_id) - pos) % length / self.visibility_length)
                                 for pos in
                                 lane_followers_pos]
                        f_pos.insert(max_lanes - 1, -1.)
                        l_pos = [(pos - self.k.vehicle.get_x_by_id(rl_id)) % length / self.visibility_length
                                 for pos in
                                 lane_leaders_pos]
                        l_pos.insert(max_lanes - 1, +1.)
                        lanes = [1.]

                    else:
                        f_sp = [(speed - rl_speed) / max_speed
                                for speed in lane_followers_speed]
                        l_sp = [(speed - rl_speed) / max_speed
                                for speed in lane_leaders_speed]
                        f_pos = [-((self.k.vehicle.get_x_by_id(rl_id) - pos) % length / self.visibility_length)
                                 for pos in lane_followers_pos]
                        l_pos = [(pos - self.k.vehicle.get_x_by_id(rl_id)) % length / self.visibility_length
                                 for pos in lane_leaders_pos]
                        lanes = [0.5]

                    rl_sp = [rl_speed / max_speed]
                    positions = l_pos + f_pos
                    speeds = rl_sp + l_sp + f_sp
                    density = lane_leader_density
                    lane_existence = [1., 1., 1.]

                    # if none value
                    for i in range(len(speeds)):
                        if speeds[i] > 1.0 or speeds[i] < -1.0:
                            speeds[i] = -1.0
                            positions[i-1] = -1.0

                        else:
                            pass

                    # target_velo = [tv / max_speed]
                    for i in range(len(positions)):
                        if -1. <= positions[i] <= 1.:
                            pass
                        else:
                            positions[i] = -1.

                    observation = np.array(speeds + positions + density + lane_existence)
                    obs.update({rl_id: observation})

        return obs

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        for veh_id in self.visible:
            self.k.vehicle.set_observed(veh_id)