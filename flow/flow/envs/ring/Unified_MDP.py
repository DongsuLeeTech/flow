from asyncore import write
from matplotlib.pyplot import sca
from sympy import N
from flow.envs.ring.accel import AccelEnv
from flow.envs.ring.lane_change_accel import LaneChangeAccelEnv
from flow.core import MA_bottle_ring_rewards as rewards

from gym.spaces.box import Box
from gym.spaces.tuple import Tuple
from gym.spaces.multi_discrete import MultiDiscrete

import numpy as np

from collections import defaultdict
from pprint import pprint

#bottleneck reward import
from flow.core import rewards as bottle_reward


###################################
import numpy as np
from gym.spaces.box import Box
import random
from scipy.optimize import fsolve
from copy import deepcopy

from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.envs.multiagent.base import MultiEnv
from flow.envs.ring.wave_attenuation import v_eq_max_function

import time

ADDITIONAL_ENV_PARAMS = {
    "max_accel": 1,
    "max_decel": -1,
    "lane_change_duration": 5,
    "target_velocity": 12.5,
    'sort_vehicles': False
}

class MADLCAccelPOEnv(MultiEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        self.j = 0
        self.cul_reward = 0

    @property
    def action_space(self):
        """See class definition."""

        max_decel = self.env_params.additional_params["max_decel"]
        max_accel = self.env_params.additional_params["max_accel"]
        lb = [-abs(max_decel), -1]
        ub = [max_accel, 1]
        return Box(np.array(lb), np.array(ub), dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=-2,
            high=2,
            shape = (5*5 + 5,),
            dtype=np.float32)

    def get_state(self):
        """See class definition."""
        obs = {}
        length = self.k.network.length()
        # rl 차량 id
        rl_ids = self.k.vehicle.get_rl_ids()
        if self.my_pos is None:
            self.my_pos = [None]*len(rl_ids)
            self.now_lane_nums = [None]*len(rl_ids)
            self.prev_lane_nums = [None]*len(rl_ids)
            self.my_lane = [None]*len(rl_ids)
            self.my_speed = [None]*len(rl_ids)

            self.prev_my_lane = [None]*len(rl_ids)

        for i, rl in enumerate(rl_ids):
        # 현재 rl 차량 edge 이름
            my_edge = self.k.vehicle.get_edge(rl)

            self.my_lane[i] = self.k.vehicle.get_lane(rl)
            self.now_lane_nums[i] = self.k.network.num_lanes(my_edge)
            self.my_pos[i] = self.k.vehicle.get_x_by_id(rl)
            self.my_speed[i] = self.k.vehicle.get_speed(rl)

            is_shrink = [-1, -1, -1]

            # visibility length 끝의 leader edge
            if self.my_pos[i] > length - self.visibility_length:
                visible_leader_edge = self.k.network.get_edge(self.my_pos[i]+self.visibility_length-length)[0]
            else:
                visible_leader_edge = self.k.network.get_edge(self.my_pos[i]+self.visibility_length)[0]

            if ":" in visible_leader_edge:
                next_lanes = self.k.network.num_lanes(visible_leader_edge[1:-2])
                junction_node = visible_leader_edge
            else:
                next_lanes = self.k.network.num_lanes(visible_leader_edge)
                junction_node = ':' + visible_leader_edge + '_0'

            # merge 구간과의 거리가 visibility length보다 작을 때 observation
            if (next_lanes != self.now_lane_nums[i]) and (':' not in my_edge):
                distance_junction = max(0, self.k.network.get_x(junction_node, 0) - self.my_pos[i])
                is_shrink = [self.now_lane_nums[i]/4, next_lanes/4, distance_junction/self.visibility_length]

            else:
                is_shrink = [self.now_lane_nums[i]/4, next_lanes/4, -1]

            # print(is_shrink)
            all_ids = np.array(self.k.vehicle.get_human_ids())

            all_pos = np.array(self.k.vehicle.get_x_by_id(all_ids))
            if self.my_pos[i] < self.visibility_length:
                all_pos[all_pos > length-self.visibility_length] -= length

            elif self.my_pos[i] > length-self.visibility_length:
                all_pos[all_pos < self.visibility_length] += length

            relative_pos = all_pos - self.my_pos[i]


            leader_dict = self.get_LF_dict(all_ids, relative_pos, 'L')
            follower_dict = self.get_LF_dict(all_ids, relative_pos, 'F')

            leader_pos_obs, leader_speed_obs, lane_density_obs = self.get_LF_obs(leader_dict, 'L', visible_leader_edge, i)
            folower_pos_obs, folower_speed_obs, _ = self.get_LF_obs(follower_dict, 'F', visible_leader_edge, i)


            lane_criterion = self.my_lane[i] / (self.now_lane_nums[i]-1)
            if lane_criterion == 0 or lane_criterion==1:
                norm_lane_pos = lane_criterion
            else:
                norm_lane_pos = 0.5


            Speed_OBS = leader_speed_obs + folower_speed_obs + [self.my_speed[i]/self.max_speed]
            Pos_OBS = leader_pos_obs + folower_pos_obs + [norm_lane_pos]
            Lane_Density_OBS = lane_density_obs
            Shrink_OBS = is_shrink

            test_obs = Speed_OBS + Pos_OBS + Lane_Density_OBS + Shrink_OBS

            observation = np.array(test_obs)
            obs.update({rl: observation})

        return obs

    def _apply_rl_actions(self, rl_actions):
        """Split the accelerations by ring."""
        file_name5 = "./result/curr_ring/exploaration/exploration.csv"
        if rl_actions:
            rl_ids = list(rl_actions.keys())
            accel = []
            direction = []
            for i in rl_actions.values():
                accel.append(i[::2])
                lc = i[1::2]

                if lc <= -0.333:
                    lc = -1
                elif lc >= 0.333:
                    lc = 1
                else:
                    lc = 0
                direction.append(lc)

            self.k.vehicle.apply_acceleration(rl_ids, accel)
            self.k.vehicle.apply_lane_change(rl_ids, direction=direction)


    def compute_reward(self, actions, **kwargs):

        if kwargs['fail']:
            print('사고발생')

        reward_dict = {}
        rls = self.k.vehicle.get_rl_ids()

        rl_des = self.initial_config.reward_params.get('rl_desired_speed', 0)
        rl_action_p = self.initial_config.reward_params.get('rl_action_penalty', 0)
        uns4IDM_p = self.initial_config.reward_params.get('uns4IDM_penalty', 0)
        mlp = self.initial_config.reward_params.get('meaningless_penalty', 0)
        cr = self.initial_config.reward_params.get('collective_reward', 0)

        alpha = self.initial_config.reward_params.get('alpha', 0)
        beta = self.initial_config.reward_params.get('beta', 0)

        rl_des = self.initial_config.reward_params.get('rl_desired_speed', 0)
        ####reward 시작####

        vel_reward = []
        lc_reward = []
        usd_reward = []

        for i, rl in enumerate(rls):
            my_edge = self.k.vehicle.get_edge(rl)
            my_lane = self.k.vehicle.get_lane(rl)
            now_lane_nums = self.k.network.num_lanes(my_edge)
            my_pos = self.k.vehicle.get_x_by_id(rl)

            if now_lane_nums == self.max_lanes:
                my_lane = my_lane
            elif now_lane_nums < self.max_lanes:
                my_lane = my_lane+1

            all_ids = np.array(self.k.vehicle.get_ids())
            others_ids = all_ids[all_ids != rl]
            others_lanes = np.array(self.k.vehicle.get_lane(others_ids))

            others_edges = np.array(self.k.vehicle.get_edge(others_ids))
            others_lane_nums = np.array([self.k.network.num_lanes(edge) for edge in others_edges])

            others_lanes[others_lane_nums==2] +=1

            same_lane_ids = others_ids[others_lanes == my_lane]
            if len(same_lane_ids) == 0:
                real_leader_headway = self.length
                real_follower_tail_way = self.length
                real_leader_speed = 0
                real_follower_speed = 0

            else:
                #sli means same_lane_ids
                sli_position = np.array(self.k.vehicle.get_x_by_id(same_lane_ids))

                leader_pose_list = sli_position[sli_position>my_pos]
                follwer_pos_list = sli_position[sli_position<my_pos]

                leader_pos = np.min(leader_pose_list) if len(leader_pose_list) !=0 else np.min(sli_position)
                follower_pos = np.max(follwer_pos_list) if len(follwer_pos_list) !=0 else np.max(sli_position)


                leader_id = same_lane_ids[sli_position==leader_pos]
                real_leader_headway = leader_pos-my_pos if leader_pos>my_pos else leader_pos + self.length - my_pos
                real_leader_speed = self.k.vehicle.get_speed(leader_id)[0]

                follower_id = same_lane_ids[sli_position==follower_pos]
                real_follower_tail_way = my_pos-follower_pos if my_pos>follower_pos else my_pos + self.length - follower_pos
                real_follower_speed = self.k.vehicle.get_speed(follower_id)[0]


            rwds = defaultdict(int)

            if rl_des:
                if self.k.vehicle.get_speed(rl) >= 0.:
                    r = rewards.rl_desired_speed(self, rl)
                    rwds['rl_desired_speed'] += r
                    vel_reward.append(r)
                else:
                    return 0.

            if uns4IDM_p:
                r = rewards.unsafe_distance_penalty4IDM(self, self.prev_my_lane[i], my_lane, real_follower_tail_way, real_follower_speed, self.prev_lane_nums[i], now_lane_nums, rl)
                rwds['uns4IDM_penalty'] += r
                usd_reward.append(r)

            if (mlp) and (self.prev_my_lane[i] is not None):
                headway = real_leader_headway
                pen = rewards.meaningless_penalty(self, self.prev_my_lane[i], my_lane, \
                                            self.prev_leader_headway[i], headway, \
                                            self.prev_lane_nums[i], now_lane_nums, rl)
                rwds['meaningless_penalty'] += pen
                self.prev_leader_headway[i] = headway
                lc_reward.append(pen)

            if rl_action_p:
                pen = rewards.rl_action_penalty(self, actions, rl)
                rwds['rl_action_penalty'] += pen

            individual_rwd = rwds['rl_desired_speed']
            penalty = rwds['uns4IDM_penalty']+rwds['rl_action_penalty']+rwds['meaningless_penalty']

            rwd = (alpha * individual_rwd) + penalty
            self.prev_my_lane[i] = my_lane
            self.prev_lane_nums[i] = now_lane_nums
            reward_dict.update({rl: np.array(rwd)})

        return reward_dict

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        for rl in self.k.vehicle.get_rl_ids():
            self.k.vehicle.set_observed(rl)

class MADLCAccelPOEnv(MADLCAccelPOEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        self.length = self.k.network.length()

        self.num_lanes = []
        self.num_lanes = np.array(self.num_lanes)
        self.max_lanes = np.max(self.num_lanes)

        self.visibility_length = 30
        self.bottleneck_length = 20
        self.observable_lanes = 5
        self.vehicle_length = 5
        self.max_speed = self.k.network.max_speed()
        self.visible = []

        numrl = self.k.vehicle.get_rl_ids()

        self.my_pos = None
        self.now_lane_nums = None
        self.prev_lane_nums = None
        self.my_lane = None
        self.my_speed = None

        self.prev_leader_headway = self.visibility_length
        self.prev_my_lane = None

        self.ts = 0

    @property
    def action_space(self):
        max_decel = self.env_params.additional_params["max_decel"]
        max_accel = self.env_params.additional_params["max_accel"]
        lb = [-abs(max_decel), -1]
        ub = [max_accel, 1]
        return Box(np.array(lb), np.array(ub), dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=-2,
            high=2,
            shape=(5*5 + 5,),
            dtype=np.float32)

    # 관측 가능 거리 내의 leader, follower 정보 lane번호 별로 정리된 dictionary 리턴
    def get_LF_dict(self, all_ids, relative_pos, type):
        if type == 'L':
            lf_ids = all_ids[(relative_pos >= 0) & (relative_pos <= self.visibility_length)]
            lf_pos = relative_pos[(relative_pos >= 0) & (relative_pos <= self.visibility_length)]
        elif type == 'F':
            lf_ids = all_ids[(relative_pos < 0) & (relative_pos >= -self.visibility_length)]
            lf_pos = relative_pos[(relative_pos < 0) & (relative_pos >= -self.visibility_length)]

        lf_edge = np.array(self.k.vehicle.get_edge(lf_ids))
        lf_lanes = np.array(self.k.vehicle.get_lane(lf_ids))

        lf_dict = dict()
        for i in range(self.max_lanes):
            lf_dict[i] = {'id' : [], 'pos' : [], 'speed': []}

        for i in range(len(lf_edge)):
            if lf_edge[i] in self.lane_criterion:
                lf_lanes[i] +=1

            lf_dict[lf_lanes[i]]['id'].append(lf_ids[i])
            lf_dict[lf_lanes[i]]['pos'].append(lf_pos[i])
            speed = self.k.vehicle.get_speed(lf_ids[i])

            try:
                lf_dict[lf_lanes[i]]['speed'].append(self.k.vehicle.get_speed(lf_ids[i]))
            except:
                f = open("./dict_error.txt", 'a')
                f.write(f'레인 : {i},\n, 바로 append하려는 speed : {self.k.vehicle.get_speed(lf_ids[i])}\n 미리 계산한 speed : {speed}\n\n\n')
                f.close()
                lf_dict[lf_lanes[i]]['speed'].append(speed)

        return lf_dict

    # leader, follower dictionary를 통해 oberservation 계산(기준 dim : max_lanes)
    def get_LF_obs(self, lf_dict, type, leader_edge, rl_id):
        pos_obs = [-1]*self.max_lanes
        speed_obs = [-1]*self.max_lanes
        lane_density_obs = [-1]*self.max_lanes

        for lane in range(self.max_lanes):
            ids = np.array(lf_dict[lane]['id'])

            if len(ids)!=0:
                pos = np.array(lf_dict[lane]['pos'])
                speed = np.array(lf_dict[lane]['speed'])

                if type == 'L':
                    relative_pos = np.min(pos)
                elif type == 'F':
                    relative_pos = np.max(pos)

                relative_speed = speed[pos==relative_pos] - self.my_speed[rl_id]
                real_leader_follwer = ids[pos==relative_pos]

                # 최대 밀도 : 관측가능거리/(차량길이+최소안전거리)
                max_density = self.visibility_length//(self.vehicle_length+5)

                # 관측 가능 거리 이후 레인개수가 변경되면
                if self.now_lane_nums[rl_id] != self.k.network.num_lanes(leader_edge):
                    # 좁아지는 도로 즉 양 끝 도로에 대해서
                    if (lane == 0) or (lane == self.max_lanes-1):
                        # 밀도 계산 거리는 레인 줄어들기 직전 거리까지만 계산
                        distance_other_lanes_road = self.k.network.get_x(leader_edge, 0) - self.my_pos[rl_id]

                        try:
                            if self.now_lane_nums[rl_id] > self.k.network.num_lanes(leader_edge):
                                pos_criterion = distance_other_lanes_road
                                max_density = pos_criterion//(self.vehicle_length+5)
                                lane_density = min(1, len(ids[pos <= pos_criterion]) / max_density)

                            elif self.now_lane_nums[rl_id] < self.k.network.num_lanes(leader_edge):
                                pos_criterion = self.visibility_length - distance_other_lanes_road
                                max_density = pos_criterion//(self.vehicle_length+5)
                                lane_density = min(1, len(ids[pos <= pos_criterion]) / max_density)

                        except ZeroDivisionError:
                            lane_density = 0.

                    else:
                        lane_density = len(ids)/max_density

                else:
                    lane_density = len(ids)/max_density


                pos_obs[lane] = relative_pos/self.visibility_length
                speed_obs[lane] = relative_speed[0]/self.max_speed
                lane_density_obs[lane] = lane_density

            else:
                pos_obs[lane] = -1.
                speed_obs[lane] = -1.
                lane_density_obs[lane] = 0.

        pos_obs = self.convert_observation_form(pos_obs, -1., rl_id, self.my_lane[rl_id])
        speed_obs = self.convert_observation_form(speed_obs, -1., rl_id, self.my_lane[rl_id])

        if type == 'L':
            lane_density_obs = self.convert_observation_form(lane_density_obs, 0., rl_id, now_lane=None)
        elif type == 'F':
            lane_density_obs = None
        return pos_obs, speed_obs, lane_density_obs

    # max lane 개수에 맞게 형성된 observation형태를 최종 형태로 변환
    def convert_observation_form(self, raw_list, empty_value, rl_id, now_lane = None):
        return_list = [empty_value] * self.observable_lanes
        mid_index = 2
        if now_lane is None:
            now_lane = mid_index

        else:
            if self.now_lane_nums[rl_id] != self.max_lanes:
                now_lane += (self.max_lanes-self.now_lane_nums[rl_id])//2

        return_list[0 : self.max_lanes] = raw_list

        scaling = now_lane - mid_index
        # print(return_list)
        if scaling !=0:
            lower_index = max(0, scaling)
            upper_index = min(self.max_lanes, self.max_lanes+scaling+1)

            return_list = [empty_value]*abs(min(0, scaling)) + return_list[lower_index:upper_index]
            return_list +=[empty_value] * (self.observable_lanes - len(return_list))

        return return_list
    def get_state(self):
        """See class definition."""
        obs = {}
        length = self.k.network.length()
        # rl 차량 id
        rl_ids = self.k.vehicle.get_rl_ids()
        if self.my_pos is None:
            self.my_pos = [None]*len(rl_ids)
            self.now_lane_nums = [None]*len(rl_ids)
            self.prev_lane_nums = [None]*len(rl_ids)
            self.my_lane = [None]*len(rl_ids)
            self.my_speed = [None]*len(rl_ids)
            self.prev_leader_headway = [self.visibility_length]*len(rl_ids)

            self.prev_my_lane = [None]*len(rl_ids)

        for i, rl in enumerate(rl_ids):
        # 현재 rl 차량 edge 이름
            my_edge = self.k.vehicle.get_edge(rl)


            # rl 관련 정보 갱신
            self.my_lane[i] = self.k.vehicle.get_lane(rl)
            self.now_lane_nums[i] = self.k.network.num_lanes(my_edge)
            self.my_pos[i] = self.k.vehicle.get_x_by_id(rl)
            self.my_speed[i] = self.k.vehicle.get_speed(rl)

            # visibility length 끝의 leader edge
            if self.my_pos[i] > length - self.visibility_length:
                visible_leader_edge = self.k.network.get_edge(self.my_pos[i]+self.visibility_length-length)[0]
            else:
                visible_leader_edge = self.k.network.get_edge(self.my_pos[i]+self.visibility_length)[0]


            if ":" in visible_leader_edge:
                next_lanes = self.k.network.num_lanes(visible_leader_edge[1:-2])
                junction_node = visible_leader_edge
            else:
                next_lanes = self.k.network.num_lanes(visible_leader_edge)
                junction_node = ':' + visible_leader_edge + '_0'

            # is_shrink = [현재 차선개수, 다음 차선개수, 차량의 pos 정보]
            is_shrink = [self.now_lane_nums[i]/4, next_lanes/4, self.my_pos[i]/self.length]

            all_ids = np.array(self.k.vehicle.get_ids())
            all_ids = all_ids[all_ids!=rl]

            all_pos = np.array(self.k.vehicle.get_x_by_id(all_ids))
            if self.my_pos[i] < self.visibility_length:
                all_pos[all_pos > length-self.visibility_length] -= length

            elif self.my_pos[i] > length-self.visibility_length:
                all_pos[all_pos < self.visibility_length] += length

            relative_pos = all_pos - self.my_pos[i]


            leader_dict = self.get_LF_dict(all_ids, relative_pos, 'L')
            follower_dict = self.get_LF_dict(all_ids, relative_pos, 'F')

            leader_pos_obs, leader_speed_obs, lane_density_obs = self.get_LF_obs(leader_dict, 'L', visible_leader_edge, i)
            folower_pos_obs, folower_speed_obs, _ = self.get_LF_obs(follower_dict, 'F', visible_leader_edge, i)

            lane_criterion = self.my_lane[i] / (self.now_lane_nums[i]-1)
            if lane_criterion == 0 or lane_criterion==1:
                norm_lane_pos = lane_criterion
            else:
                norm_lane_pos = 0.5

            Speed_OBS = leader_speed_obs + folower_speed_obs + [self.my_speed[i]/self.max_speed]
            Pos_OBS = leader_pos_obs + folower_pos_obs + [norm_lane_pos]
            Lane_Density_OBS = lane_density_obs
            Shrink_OBS = is_shrink

            test_obs = Speed_OBS + Pos_OBS + Lane_Density_OBS + Shrink_OBS

            observation = np.array(test_obs)
            obs.update({rl: observation})

        return obs

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        for rl in self.k.vehicle.get_rl_ids():
            self.k.vehicle.set_observed(rl)