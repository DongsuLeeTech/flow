from asyncore import write
from matplotlib.pyplot import sca
from sympy import N
from flow.envs.ring.accel import AccelEnv
from flow.envs.ring.lane_change_accel import LaneChangeAccelEnv
from flow.core import MA_ring_rewards as rewards

from gym.spaces.box import Box
from gym.spaces.tuple import Tuple
from gym.spaces.multi_discrete import MultiDiscrete

import numpy as np

from collections import defaultdict
from pprint import pprint

# bottleneck reward import

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


class MARing_5vis_Env(MultiEnv):
    """POMDP version of LaneChangeAccelEnv.
        Required from env_params:
        * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
        * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
        * lane_change_duration: lane change duration for autonomous vehicles, in s
        * target_velocity: desired velocity for all vehicles in the network, in m/s
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        self.j = 0
        self.cul_reward = 0
        self.vehicle_length = 4.45
        self.crash = False

        self.prev_lc_action = None
        self.max_acc = 5.4
        self.des_vel = 13.686  # It is nessesary when you want to apply adaptive reward model

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
            low=-1,
            high=1,
            shape=(5 * 5 + 5,),
            dtype=np.float32)

    def get_state(self):
        """See class definition."""
        obs = {}
        length = self.k.network.length()
        # rl 차량 id
        rl_ids = self.k.vehicle.get_rl_ids()
        if self.my_pos is None:
            self.my_pos = [None] * len(rl_ids)
            self.now_lane_nums = [None] * len(rl_ids)
            self.prev_lane_nums = [None] * len(rl_ids)
            self.my_lane = [None] * len(rl_ids)
            self.my_speed = [None] * len(rl_ids)

            self.prev_my_lane = [None] * len(rl_ids)

        for i, rl in enumerate(rl_ids):
            # 현재 rl 차량 edge 이름
            my_edge = self.k.vehicle.get_edge(rl)

            # rl 관련 정보 갱신
            self.my_lane[i] = self.k.vehicle.get_lane(rl)
            self.now_lane_nums[i] = self.k.network.num_lanes(my_edge)
            self.my_pos[i] = self.k.vehicle.get_x_by_id(rl)
            self.my_speed[i] = self.k.vehicle.get_speed(rl)

            is_shrink = [-1, -1, -1]

            # visibility length 끝의 leader edge
            if self.my_pos[i] > length - self.visibility_length:
                visible_leader_edge = self.k.network.get_edge(self.my_pos[i] + self.visibility_length - length)[0]
            else:
                visible_leader_edge = self.k.network.get_edge(self.my_pos[i] + self.visibility_length)[0]

            if ":" in visible_leader_edge:
                next_lanes = self.k.network.num_lanes(visible_leader_edge[1:-2])
                junction_node = visible_leader_edge
            else:
                next_lanes = self.k.network.num_lanes(visible_leader_edge)
                junction_node = ':' + visible_leader_edge + '_0'

            # merge 구간과의 거리가 visibility length보다 작을 때 observation
            if (next_lanes != self.now_lane_nums[i]) and (':' not in my_edge):
                distance_junction = max(0, self.k.network.get_x(junction_node, 0) - self.my_pos[i])
                is_shrink = [self.now_lane_nums[i] / 4, next_lanes / 4, distance_junction / self.visibility_length]

            else:
                is_shrink = [self.now_lane_nums[i] / 4, next_lanes / 4, -1]

            all_ids = np.array(self.k.vehicle.get_human_ids())

            all_pos = np.array(self.k.vehicle.get_x_by_id(all_ids))
            if self.my_pos[i] < self.visibility_length:
                all_pos[all_pos > length - self.visibility_length] -= length

            elif self.my_pos[i] > length - self.visibility_length:
                all_pos[all_pos < self.visibility_length] += length

            relative_pos = all_pos - self.my_pos[i]

            leader_dict = self.get_LF_dict(all_ids, relative_pos, 'L')
            follower_dict = self.get_LF_dict(all_ids, relative_pos, 'F')

            leader_pos_obs, leader_speed_obs, lane_density_obs = self.get_LF_obs(leader_dict, 'L', visible_leader_edge,
                                                                                 i)
            folower_pos_obs, folower_speed_obs, _ = self.get_LF_obs(follower_dict, 'F', visible_leader_edge, i)

            lane_criterion = self.my_lane[i] / (self.now_lane_nums[i] - 1)
            if lane_criterion == 0 or lane_criterion == 1:
                norm_lane_pos = lane_criterion
            else:
                norm_lane_pos = 0.5

            Speed_OBS = leader_speed_obs + folower_speed_obs + [self.my_speed[i] / self.max_speed]
            Pos_OBS = leader_pos_obs + folower_pos_obs + [norm_lane_pos]
            Lane_Density_OBS = lane_density_obs
            Shrink_OBS = is_shrink

            test_obs = Speed_OBS + Pos_OBS + Lane_Density_OBS + Shrink_OBS

            observation = np.array(test_obs)
            obs.update({rl: observation})
        return obs

    def _apply_rl_actions(self, rl_actions):
        """Split the accelerations by ring."""

        if rl_actions:
            rl_ids = list(rl_actions.keys())

            # if self.lc_action is None:
            #     self.lc_action = [None]*len(rl_ids)
            accel = []
            direction = []

            for i in rl_actions.values():
                # bmil edit 11/13
                accel.append(i[::2] * self.max_acc)
                lc = i[1::2]

                if lc <= -0.333:
                    lc = -1
                elif lc >= 0.333:
                    lc = 1
                else:
                    lc = 0
                direction.append(lc)

            self.prev_lc_action = np.array(direction)
            self.k.vehicle.apply_acceleration(rl_ids, accel)
            self.k.vehicle.apply_lane_change(rl_ids, direction=direction)

    def compute_reward(self, actions, **kwargs):

        reward_dict = {}
        rls = self.k.vehicle.get_rl_ids()
        rl_des = self.initial_config.reward_params.get('rl_desired_speed', 0)
        rl_action_p = self.initial_config.reward_params.get('rl_action_penalty', 0)
        uns4IDM_p = self.initial_config.reward_params.get('uns4IDM_penalty', 0)
        leader_uns_p = self.initial_config.reward_params.get('leader_uns_penalty', 0)
        mlp = self.initial_config.reward_params.get('meaningless_penalty', 0)
        cr = self.initial_config.reward_params.get('collective_reward', 0)

        alpha = self.initial_config.reward_params.get('alpha', 0)
        beta = self.initial_config.reward_params.get('beta', 0)

        default_reward = self.initial_config.reward_params.get('default_reward', 0)
        accident_p = self.initial_config.reward_params.get('accident_penalty', 0)

        ####reward 시작####

        vel_reward = []
        lc_reward = []
        usd_reward = []

        # bmil edit 2023/01/06

        for i, rl in enumerate(rls):
            if self.crash[i] == True:
                rwd = default_reward + accident_p
            else:
                my_edge = self.k.vehicle.get_edge(rl)
                my_lane = self.k.vehicle.get_lane(rl)
                now_lane_nums = self.k.network.num_lanes(my_edge)
                my_pos = self.k.vehicle.get_x_by_id(rl)
                if now_lane_nums == self.max_lanes:
                    my_lane = my_lane
                elif now_lane_nums < self.max_lanes:
                    my_lane = my_lane + 1

                remain_rl = list(np.array(rls)[self.crash == False])
                human_ids = np.array(self.k.vehicle.get_human_ids())

                human_lane = np.array(self.k.vehicle.get_lane(human_ids))
                remain_human = list(human_ids[human_lane >= 0])
                all_ids = np.array(remain_rl + remain_human)

                others_ids = all_ids[all_ids != rl]
                others_lanes = np.array(self.k.vehicle.get_lane(others_ids))

                others_edges = np.array(self.k.vehicle.get_edge(others_ids))
                others_lane_nums = np.array([self.k.network.num_lanes(edge) for edge in others_edges])

                others_lanes[others_lane_nums == 2] += 1

                same_lane_ids = others_ids[others_lanes == my_lane]
                if len(same_lane_ids) == 0:
                    real_leader_headway = self.length
                    real_follower_tail_way = self.length
                    real_leader_speed = 0
                    real_follower_speed = 0

                else:
                    sli_position = np.array(self.k.vehicle.get_x_by_id(same_lane_ids))

                    leader_pose_list = sli_position[sli_position > my_pos]
                    follwer_pos_list = sli_position[sli_position < my_pos]

                    leader_pos = np.min(leader_pose_list) if len(leader_pose_list) != 0 else np.min(sli_position)
                    follower_pos = np.max(follwer_pos_list) if len(follwer_pos_list) != 0 else np.max(sli_position)

                    leader_id = same_lane_ids[sli_position == leader_pos]

                    # bmil edit 01/31
                    if (self.my_edge[i] == self.shring_junction) and np.any(self.num_lanes != self.max_lanes):

                        ids_in_sj = np.array(self.k.vehicle.get_ids_by_edge(self.my_edge))
                        ids_in_sj = ids_in_sj[ids_in_sj != rl]
                        sj_lanes = np.array(self.k.vehicle.get_lane(ids_in_sj)) // 2
                        sj_ids = ids_in_sj[sj_lanes == self.my_lane[i]]

                        rl_pos = np.array(self.k.vehicle.get_x_by_id(rl))
                        sj_pos = np.array(self.k.vehicle.get_x_by_id(sj_ids))
                        sj_leader_pos = sj_pos[sj_pos >= rl_pos]
                        sj_leader_ids = sj_ids[sj_pos >= rl_pos]

                        if len(sj_leader_pos) > 0:
                            leader_id = sj_leader_ids[sj_leader_pos == np.min(sj_leader_pos)]
                            leader_pos = self.k.vehicle.get_x_by_id(leader_id)[0]

                    real_leader_headway = leader_pos - my_pos if leader_pos > my_pos else leader_pos + self.length - my_pos
                    real_leader_speed = self.k.vehicle.get_speed(leader_id)[0]
                    follower_id = same_lane_ids[sli_position == follower_pos]
                    real_follower_tail_way = my_pos - follower_pos if my_pos > follower_pos else my_pos + self.length - follower_pos
                    real_follower_speed = self.k.vehicle.get_speed(follower_id)[0]

                rwds = defaultdict(int)
                # 바로 아랫 줄을 제거하거나 exponent 로 바꾸는 편 좋을듯.
                # personal_reward

                if self.crash[i] == False:
                    if rl_des:
                        if self.k.vehicle.get_speed(rl) >= 0.:
                            r = rewards.rl_desired_speed(self, rl)
                            rwds['rl_desired_speed'] += r
                            vel_reward.append(r)
                        else:
                            return 0.

                    if uns4IDM_p:
                        # state 기반
                        tailway = real_follower_tail_way
                        follower_speed = real_follower_speed

                        if tailway > self.visibility_length:  # mim 함수 안쓰는 이유 -> 실제 tailway가 visibility_length와 같은경우 안전거리 고려하기 위해
                            tailway = self.visibility_length
                            follower_speed = -1

                        r = rewards.unsafe_distance_penalty4IDM(self, self.prev_my_lane[i], my_lane, tailway,
                                                                follower_speed, \
                                                                self.prev_lane_nums[i], now_lane_nums, rl)
                        rwds['uns4IDM_penalty'] += r
                        usd_reward.append(r)

                    # bmil edit 01/18
                    if leader_uns_p:
                        # state 기반
                        headway = real_leader_headway
                        leader_speed = real_leader_speed
                        rl_speed = self.k.vehicle.get_speed(rl)

                        if headway > self.visibility_length:
                            headway = self.visibility_length
                            leader_speed = -1  # 상대속도 양수이면 -> leader가 더 빠르면, penalty는 어쩌피 0이기 때문
                        pen = rewards.leader_unsafe_distance_penalty(self, headway, rl_speed, leader_speed,
                                                                     self.vehicle_length)
                        rwds['leader_uns_penalty'] += pen
                    if (mlp) and (self.prev_my_lane[i] is not None):
                        headway = real_leader_headway  # state 기반

                        headway = min(self.visibility_length, headway)

                        pen = rewards.meaningless_penalty(self, self.prev_my_lane[i], my_lane, \
                                                          self.prev_leader_headway[i], headway, \
                                                          self.prev_lane_nums[i], now_lane_nums, rl,
                                                          self.visibility_length)
                        rwds['meaningless_penalty'] += pen
                        self.prev_leader_headway[i] = headway

                        lc_reward.append(pen)

                    if rl_action_p:
                        pen = rewards.rl_action_penalty(self, actions, rl)
                        rwds['rl_action_penalty'] += pen

                    individual_rwd = rwds['rl_desired_speed']
                    penalty = rwds['uns4IDM_penalty'] + rwds['leader_uns_penalty'] + rwds['rl_action_penalty'] + rwds[
                        'meaningless_penalty']

                    # edit 1/16
                    rwd = default_reward + (alpha * individual_rwd) + penalty
                self.prev_my_lane[i] = my_lane
                self.prev_lane_nums[i] = now_lane_nums
                self.prev_pos[i] = self.my_pos[i]

            reward_dict.update({rl: np.array(rwd)})

        return reward_dict

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        for rl in self.k.vehicle.get_rl_ids():
            self.k.vehicle.set_observed(rl)


class MADRingLCPO_5vis_Env(MARing_5vis_Env):
    """POMDP version of LaneChangeAccelEnv.
        Required from env_params:
        * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
        * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
        * lane_change_duration: lane change duration for autonomous vehicles, in s
        * target_velocity: desired velocity for all vehicles in the network, in m/s
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        ## simulation 초기 고정 파라미터
        self.length = self.k.network.length()
        self.edge_list = self.k.network.get_edge_list() + self.k.network.get_junction_list()
        self.num_lanes = []

        for edge_num in (self.edge_list):
            self.num_lanes.append(self.k.network.num_lanes(edge_num))

        self.edge_list = np.array(self.edge_list)
        self.num_lanes = np.array(self.num_lanes)
        self.max_lanes = np.max(self.num_lanes)

        self.lane_criterion = self.edge_list[self.num_lanes != self.max_lanes]

        self.visibility_length = 60
        self.bottleneck_length = 20
        self.base_obs_lane = 5  # 기본 5개 볼수 있다고 가정
        self.observable_lanes = 5  # 실제 visibile lane 수
        self.vehicle_length = 4.45
        self.max_speed = self.k.network.max_speed()
        self.visible = []
        self.obs_space_shape = self.observable_lanes * 6 + 1

        ## simulation 동안 계속 갱신될 파라미터
        numrl = self.k.vehicle.get_rl_ids()

        self.my_pos = None
        self.now_lane_nums = None
        self.prev_lane_nums = None
        self.my_lane = None
        self.my_speed = None
        self.my_edge = None
        ## reward 관련
        self.prev_leader_headway = self.visibility_length
        self.prev_my_lane = None
        self.prev_pos = None
        self.ts = 0
        self.shring_junction = ':top_0'

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
            low=-1,
            high=1,
            shape=(self.obs_space_shape,),
            dtype=np.float32)

    # 관측 가능 거리 내의 leader, follower 정보 lane번호 별로 정리된 dictionary 리턴
    def get_LF_dict(self, all_ids, relative_pos, type):
        if type == 'L':  # leader
            lf_ids = all_ids[(relative_pos >= 0) & (relative_pos <= self.visibility_length)]
            lf_pos = relative_pos[(relative_pos >= 0) & (relative_pos <= self.visibility_length)]
        elif type == 'F':  # follower
            lf_ids = all_ids[(relative_pos < 0) & (relative_pos >= -self.visibility_length)]
            lf_pos = relative_pos[(relative_pos < 0) & (relative_pos >= -self.visibility_length)]

        lf_edge = np.array(self.k.vehicle.get_edge(lf_ids))
        lf_lanes = np.array(self.k.vehicle.get_lane(lf_ids))

        lf_dict = dict()
        for i in range(self.max_lanes):
            lf_dict[i] = {'id': [], 'pos': [], 'speed': []}

        for i in range(len(lf_edge)):
            if lf_edge[i] in self.lane_criterion:
                lf_lanes[i] += 1

            try:
                lf_dict[lf_lanes[i]]['id'].append(lf_ids[i])
                lf_dict[lf_lanes[i]]['pos'].append(lf_pos[i])
                speed = self.k.vehicle.get_speed(lf_ids[i])
                lf_dict[lf_lanes[i]]['speed'].append(speed)
            except:
                print('lf_dict', lf_dict)
                print('현재 id', i, lf_ids[i])
                print('현재 pos', i, lf_pos[i])
                print('현재 speed', self.k.vehicle.get_speed(lf_ids[i]))
                print(self.crash)

                # time.sleep(0)
                lf_dict[lf_lanes[i]]['id'].append(lf_ids[i])
                lf_dict[lf_lanes[i]]['pos'].append(lf_pos[i])
                speed = self.k.vehicle.get_speed(lf_ids[i])
                lf_dict[lf_lanes[i]]['speed'].append(speed)

        return lf_dict

    # leader, follower dictionary를 통해 oberservation 계산(기준 dim : max_lanes)
    def get_LF_obs(self, lf_dict, type, leader_edge, rl_id):
        pos_obs = [-1] * self.max_lanes
        speed_obs = [-1] * self.max_lanes
        lane_density_obs = [-1] * self.max_lanes

        for lane in range(self.max_lanes):
            ids = np.array(lf_dict[lane]['id'])

            if len(ids) != 0:
                pos = np.array(lf_dict[lane]['pos'])
                speed = np.array(lf_dict[lane]['speed'])

                if type == 'L':
                    relative_pos = np.min(pos)
                elif type == 'F':
                    relative_pos = np.max(pos)

                relative_speed = speed[pos == relative_pos] - self.my_speed[rl_id]
                real_leader_follwer = ids[pos == relative_pos]

                # 최대 밀도 : 관측가능거리/(차량길이+최소안전거리)
                max_density = self.visibility_length // (self.vehicle_length)

                # 관측 가능 거리 이후 레인개수가 변경되면
                if self.now_lane_nums[rl_id] != self.k.network.num_lanes(leader_edge):
                    # 좁아지는 도로 즉 양 끝 도로에 대해서
                    if (lane == 0) or (lane == self.max_lanes - 1):
                        # 밀도 계산 거리는 레인 줄어들기 직전 거리까지만 계산
                        distance_other_lanes_road = self.k.network.get_x(leader_edge, 0) - self.my_pos[rl_id]

                        try:
                            if self.now_lane_nums[rl_id] > self.k.network.num_lanes(leader_edge):
                                pos_criterion = distance_other_lanes_road
                                max_density = pos_criterion // (self.vehicle_length)
                                lane_density = min(1, len(ids[pos <= pos_criterion]) / max_density)

                            elif self.now_lane_nums[rl_id] < self.k.network.num_lanes(leader_edge):
                                pos_criterion = self.visibility_length - distance_other_lanes_road
                                max_density = pos_criterion // (self.vehicle_length)
                                lane_density = min(1, len(ids[pos <= pos_criterion]) / max_density)

                        except ZeroDivisionError:
                            lane_density = 0.

                    else:
                        lane_density = len(ids) / max_density

                else:
                    lane_density = len(ids) / max_density

                pos_obs[lane] = relative_pos / self.visibility_length
                speed_obs[lane] = relative_speed[0] / self.max_speed
                lane_density_obs[lane] = lane_density

            else:

                if type == 'L':
                    pos_obs[lane] = 1.
                    speed_obs[lane] = 1.
                elif type == 'F':
                    pos_obs[lane] = -1.
                    speed_obs[lane] = -1.
                lane_density_obs[lane] = 0.

        if type == 'L':
            if (self.my_edge[rl_id] == self.shring_junction) and np.any(self.num_lanes != self.max_lanes):
                outter_criterion = pos_obs[0] <= pos_obs[1]
                inner_criterion = pos_obs[2] <= pos_obs[3]

                if outter_criterion:
                    pos_obs[1], lane_density_obs[1], speed_obs[1] = pos_obs[0], lane_density_obs[0], speed_obs[0]
                else:
                    pos_obs[0], lane_density_obs[0], speed_obs[0] = pos_obs[1], lane_density_obs[1], speed_obs[1]

                if inner_criterion:
                    pos_obs[3], lane_density_obs[3], speed_obs[3] = pos_obs[2], lane_density_obs[2], speed_obs[2]
                else:
                    pos_obs[2], lane_density_obs[2], speed_obs[2] = pos_obs[3], lane_density_obs[3], speed_obs[3]

                pos_obs[0], lane_density_obs[0], speed_obs[0] = 1., 1., 1.
                pos_obs[3], lane_density_obs[3], speed_obs[3] = 1., 1., 1.

        if self.now_lane_nums[rl_id] != self.max_lanes:
            lane_density_obs[0] = 1.
            lane_density_obs[-1] = 1.
            # print(lane_density_obs)

        if type == 'L':
            lane_density_obs = self.convert_observation_form(lane_density_obs, 1, rl_id, self.my_lane[rl_id])
            pos_obs = self.convert_observation_form(pos_obs, 1., rl_id, self.my_lane[rl_id])
            speed_obs = self.convert_observation_form(speed_obs, 1., rl_id, self.my_lane[rl_id])
        elif type == 'F':
            lane_density_obs = None
            pos_obs = self.convert_observation_form(pos_obs, -1., rl_id, self.my_lane[rl_id])
            speed_obs = self.convert_observation_form(speed_obs, -1., rl_id, self.my_lane[rl_id])

        return pos_obs, speed_obs, lane_density_obs

    # max lane 개수에 맞게 형성된 observation형태를 최종 형태로 변환
    def convert_observation_form(self, raw_list, empty_value, rl_id, now_lane=None):

        is_not_valid = len(raw_list) != self.base_obs_lane

        if self.max_lanes % 2 == 0:  # is_not_valid:
            raw_list += [empty_value] * is_not_valid

        num_padding = self.observable_lanes // 2
        return_list = [empty_value] * num_padding + raw_list + [empty_value] * num_padding

        mid_index = self.max_lanes // 2

        if now_lane is None:
            now_lane = mid_index

        else:
            if self.now_lane_nums[rl_id] != self.max_lanes:
                now_lane += (self.max_lanes - self.now_lane_nums[rl_id]) // 2
        # print(now_lane)
        scaling = now_lane - mid_index

        if scaling != 0:
            f_return_list = return_list[(mid_index + scaling):len(return_list) - (mid_index - scaling)]


        else:
            f_return_list = return_list[mid_index:-mid_index]

        f_return_list.reverse()

        return f_return_list

    def get_state(self):
        """See class definition."""
        obs = {}
        length = self.k.network.length()
        # rl 차량 id
        rl_ids = self.k.vehicle.get_rl_ids()

        crash = self.k.simulation.check_collision()

        # else:
        if self.my_pos is None:
            self.my_pos = [None] * len(rl_ids)
            self.now_lane_nums = [None] * len(rl_ids)
            self.prev_lane_nums = [None] * len(rl_ids)
            self.my_lane = [None] * len(rl_ids)
            self.my_speed = [None] * len(rl_ids)
            self.my_edge = [None] * len(rl_ids)
            self.prev_leader_headway = [self.visibility_length] * len(rl_ids)

            self.prev_my_lane = [None] * len(rl_ids)
            ##
            self.prev_pos = [None] * len(rl_ids)
            self.crash = np.array([False] * len(rl_ids))

        self.crash = np.array([False] * len(rl_ids))

        self.my_lane = np.array(self.k.vehicle.get_lane(rl_ids))
        self.my_pos = np.array(self.k.vehicle.get_x_by_id(rl_ids))
        # all_ids = np.array(self.k.vehicle.get_ids())

        if crash:
            print('사고 발생')
            fixed_obs = np.array([0.] * self.obs_space_shape)
            if np.all(self.my_lane >= 0):
                diff_pos = np.abs(np.array(self.my_pos) - np.array(self.prev_pos))
                diff_pos[diff_pos > 668] = 0  # when the agent pass through the start point
                crashed_idx = np.where(diff_pos > 10)[0]
                # print(diff_pos)
                # print('max pos로 계산')
            else:
                crashed_idx = np.where(self.my_lane < 0)[0]
                # print('traci가 사고 확인함')

            print(crashed_idx)
            for idx in crashed_idx:
                # print(crashed_idx)
                crashed_rl_id = rl_ids[idx]
                # print(crashed_rl_id,'get accidents')
                obs.update({crashed_rl_id: fixed_obs})
                self.crash[idx] = True
            # print(self.crash)
            # print(self.my_lane)

        remain_rl = list(np.array(rl_ids)[self.crash == False])
        human_ids = np.array(self.k.vehicle.get_human_ids())

        human_lane = np.array(self.k.vehicle.get_lane(human_ids))
        remain_human = list(human_ids[human_lane >= 0])

        full_ids = np.array(remain_rl + remain_human)

        for i, rl in enumerate(rl_ids):
            now_lane = self.k.vehicle.get_lane(rl)
            if self.crash[i] == False:
                # 현재 rl 차량 edge 이름
                self.my_edge[i] = self.k.vehicle.get_edge(rl)

                # rl 관련 정보 갱신
                self.my_lane[i] = now_lane
                self.now_lane_nums[i] = self.k.network.num_lanes(self.my_edge[i])
                self.my_pos[i] = self.k.vehicle.get_x_by_id(rl)
                self.my_speed[i] = self.k.vehicle.get_speed(rl)

                if self.my_edge[i] == self.shring_junction:
                    if self.my_lane[i] <= 1:
                        self.my_lane[i] = 0
                    else:
                        self.my_lane[i] = 1
                    self.now_lane_nums[i] = 2

                # print(self.my_lane[i])

                # visibility length 끝의 leader edge
                if self.my_pos[i] > length - self.visibility_length:
                    visible_leader_edge = self.k.network.get_edge(self.my_pos[i] + self.visibility_length - length)[0]
                else:
                    visible_leader_edge = self.k.network.get_edge(self.my_pos[i] + self.visibility_length)[0]

                if ":" in visible_leader_edge:
                    next_lanes = self.k.network.num_lanes(visible_leader_edge[1:-2])
                    junction_node = visible_leader_edge
                else:
                    next_lanes = self.k.network.num_lanes(visible_leader_edge)
                    junction_node = ':' + visible_leader_edge + '_0'

                # all_ids = np.array(self.k.vehicle.get_ids())
                # all_ids = all_ids[all_ids!=rl]
                all_ids = full_ids[full_ids != rl]
                # print(rl, len(all_ids))

                all_pos = np.array(self.k.vehicle.get_x_by_id(all_ids))
                if self.my_pos[i] < self.visibility_length:
                    all_pos[all_pos > length - self.visibility_length] -= length

                elif self.my_pos[i] > length - self.visibility_length:
                    all_pos[all_pos < self.visibility_length] += length

                relative_pos = all_pos - self.my_pos[i]

                leader_dict = self.get_LF_dict(all_ids, relative_pos, 'L')
                follower_dict = self.get_LF_dict(all_ids, relative_pos, 'F')

                leader_pos_obs, leader_speed_obs, lane_density_obs = self.get_LF_obs(leader_dict, 'L',
                                                                                     visible_leader_edge, i)
                folower_pos_obs, folower_speed_obs, _ = self.get_LF_obs(follower_dict, 'F', visible_leader_edge, i)

                lane_criterion = self.my_lane[i] / (self.now_lane_nums[i] - 1)
                # print(self.my_pos[i])
                ## 도로 끊기는지 여부
                W_after_edge = self.k.network.get_edge(self.my_pos[i] + self.visibility_length)[0]
                W_connect = self.k.network.num_lanes(W_after_edge)

                raw_num_lanes_after_W = [1.] * self.max_lanes

                if W_connect < self.max_lanes:
                    raw_num_lanes_after_W = [0., 1., 1., 0.]

                num_lanes_after_W = self.convert_observation_form(raw_num_lanes_after_W, 0., i, self.my_lane[i])

                # print(self.k.network.get_edge(self.my_pos[i]))
                # print(num_lanes_after_W)

                Speed_OBS = [self.my_speed[i] / self.max_speed] + leader_speed_obs + folower_speed_obs
                Pos_OBS = leader_pos_obs + folower_pos_obs
                Lane_Density_OBS = lane_density_obs
                Connect = num_lanes_after_W

                deco_my_speed = Speed_OBS[0]
                deco_lf_rel_speeds = np.array(Speed_OBS[1:4].copy())

                if np.all(deco_lf_rel_speeds == 1):
                    self.des_vel = 13.686
                    # print('아무도 없네')
                else:
                    deco_lf_speeds = (deco_lf_rel_speeds + deco_my_speed) * self.max_speed
                    deco_lf_speeds[deco_lf_speeds >= self.max_speed] = -1
                    # deco_lf_speeds[deco_lf_speeds<0] = 13.686 #follower 속력이 -임을 고려
                    # print(deco_lf_speeds)
                    adapt_des_vel = np.max(deco_lf_speeds)
                    self.des_vel = adapt_des_vel

                test_obs = Speed_OBS + Pos_OBS + Lane_Density_OBS + Connect
                observation = np.array(test_obs)
                observation = np.clip(observation, -1, 1)
                obs.update({rl: observation})

        return obs

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        for rl in self.k.vehicle.get_rl_ids():
            self.k.vehicle.set_observed(rl)