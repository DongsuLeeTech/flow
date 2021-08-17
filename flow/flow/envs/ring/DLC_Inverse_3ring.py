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


class DLCIAccelEnv3(AccelEnv):
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

        lb = [-abs(max_decel), -1] * self.initial_vehicles.num_rl_vehicles
        ub = [max_accel, 1.] * self.initial_vehicles.num_rl_vehicles
        shape = self.initial_vehicles.num_rl_vehicles + 1,
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
        rls = self.k.vehicle.get_rl_ids()
        reward = 0

        rl_des = self.initial_config.reward_params.get('rl_desired_speed', 0)
        simple_lc_p = self.initial_config.reward_params.get('simple_lc_penalty', 0)
        unsafe_p = self.initial_config.reward_params.get('unsafe_penalty', 0)
        dc3_p = self.initial_config.reward_params.get('dc3_penalty', 0)
        rl_action_p = self.initial_config.reward_params.get('rl_action_penalty', 0)
        acc_p = self.initial_config.reward_params.get('acc_penalty', 0)
        uns4IDM_p = self.initial_config.reward_params.get('uns4IDM_penalty', 0)
        mlp = self.initial_config.reward_params.get('meaningless_penalty', 0)

        rwds = defaultdict(int)

        for rl in rls:
            # 바로 아랫 줄을 제거하거나 exponent 로 바꾸는 편 좋을듯.
            if rl_des:
                if self.k.vehicle.get_speed(rl) > 0.:
                    r = rewards.rl_desired_speed(self)
                    reward += r
                    rwds['rl_desired_speed'] += r
                else:
                    return 0.

            if simple_lc_p and self.time_counter == self.k.vehicle.get_last_lc(rl):
                reward -= simple_lc_p
                rwds['simple_lc_penalty'] -= simple_lc_p

            follower = self.k.vehicle.get_follower(rl)
            leader = self.k.vehicle.get_leader(rl)
            # 수정 필요
            if leader is not None:
                if mlp:
                    pen = rewards.meaningless_penalty(self)
                    reward += pen
                    rwds['meaningless_penalty'] += pen

            if follower is not None:
                if uns4IDM_p:
                    pen = rewards.unsafe_distance_penalty4IDM(self)
                    reward += pen
                    rwds['uns4IDM_penalty'] += pen

                if acc_p:
                    pen = rewards.punish_accelerations(self, rl_actions)
                    reward += pen
                    rwds['acc_penalty'] += pen

                if unsafe_p:
                    pen = rewards.unsafe_distance_penalty(self)
                    reward += pen
                    rwds['unsafe_penalty'] += pen

                if dc3_p:
                    pen = rewards.follower_decel_penalty(self)
                    reward += pen
                    rwds['dc3_penalty'] += pen

            if rl_action_p:
                pen = rewards.rl_action_penalty(self, rl_actions)
                reward += pen
                rwds['rl_action_penalty'] += pen
        # print(rwds)
        rwd = sum(rwds.values())
        # if rwd :
        #     print('accumulative reward is negative:{}\nelements of reward:{}'.format(rwd,rwds))
        #     print(self.get_state())
        # rwd = rwd + 2
        # print(rwd)

        if self.env_params.evaluate:
            self.evaluate_rewards(rl_actions, self.initial_config.reward_params.keys())

            if self.accumulated_reward is None:
                self.accumulated_reward = defaultdict(int)
            else:
                for k in reward.keys():
                    self.accumulated_reward[k] += reward[k]

            if self.time_counter == self.env_params.horizon \
                    + self.env_params.warmup_steps - 1:
                print('=== now reward ===')
                pprint(dict(reward))
                print('=== accumulated reward ===')
                pprint(dict(self.accumulated_reward))
        return rwd

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

    # def _to_lc_action(self, rl_action):
    #     """Make direction components of rl_action to discrete"""
    #     if rl_action is None:
    #         return rl_action
    #     for i in range(1, len(rl_action), 2):
    #         if rl_action[i] <= -0.333:
    #             rl_action[i] = -1
    #         elif rl_action[i] >= 0.333:
    #             rl_action[i] = 1
    #         else:
    #             rl_action[i] = 0
    #     return rl_action

    def _apply_rl_actions(self, actions):
        # actions = self._to_lc_action(actions)
        acceleration = actions[::2]
        direction = actions[1::2]

        for i in range(len(direction)):
            if direction[i] <= -0.333:
                direction[i] = -1
            elif direction[i] >= 0.333:
                direction[i] = 1
            else:
                direction[i] = 0

        self.last_lane = self.k.vehicle.get_lane(self.k.vehicle.get_rl_ids())

        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids
                         if veh_id in self.k.vehicle.get_rl_ids()]

        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = \
            [self.time_counter <=
             self.env_params.additional_params["lane_change_duration"]
             + self.k.vehicle.get_last_lc(veh_id)
             for veh_id in sorted_rl_ids]

        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = \
            np.array([0] * sum(non_lane_changing_veh))

        self.k.vehicle.apply_acceleration(sorted_rl_ids, acc=acceleration)
        self.k.vehicle.apply_lane_change(sorted_rl_ids, direction=direction)

    def additional_command(Self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        if self.k.vehicle.num_rl_vehicles > 0:
            for veh_id in self.k.vehicle.get_human_ids():
                self.k.vehicle.set_observed(veh_id)


class DLCIAccelPOEnv3(DLCIAccelEnv3):
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
            # shape=(2 * self.initial_vehicles.num_rl_vehicles *
            #        (self.num_lanes + 5) + 2,),
            dtype=np.float32)

    def get_state(self):
        """See class definition."""
        # normalizers
        max_speed = self.k.network.max_speed()
        length = self.k.network.length()
        max_lanes = max(
            self.k.network.num_lanes(edge)
            for edge in self.k.network.get_edge_list())

        # # coef
        # rl_des = self.initial_config.reward_params.get('rl_desired_speed', 0)
        # uns4IDM_p = self.initial_config.reward_params.get('uns4IDM_penalty', 0)

        # NOTE: this works for only single agent environmnet
        rl = self.k.vehicle.get_rl_ids()[0]
        lane_followers = self.k.vehicle.get_lane_followers(rl)
        lane_leaders = self.k.vehicle.get_lane_leaders(rl)

        # Velocity of vehicles
        lane_followers_speed = self.k.vehicle.get_lane_followers_speed(rl)
        lane_leaders_speed = self.k.vehicle.get_lane_leaders_speed(rl)
        rl_speed = [self.k.vehicle.get_speed(rl)]
        for i in rl_speed:
            if i / max_speed > 1:
                rl_speed = [1.]

        # Position of Vehicles
        lane_followers_pos = [self.k.vehicle.get_x_by_id(follower) for follower in lane_followers]
        lane_leaders_pos = [self.k.vehicle.get_x_by_id(leader) for leader in lane_leaders]

        # #  Relative Lane of Vehicles
        # follower_lanes = [self.k.vehicle.get_lane(follower) for follower in lane_followers]
        # leader_lanes = [self.k.vehicle.get_lane(leader) for leader in lane_leaders]

        #  Normalization of states
        # speeds = [speed / max_speed
        #           for speed in lane_followers_speed + lane_leaders_speed + rl_speed]

        for i in range(0, max_lanes):
            # print(max_lanes)
            if self.k.vehicle.get_lane(rl) == i:
                lane_followers_speed = lane_followers_speed[max(0, i - 1):i + 2]
                lane_leaders_speed = lane_leaders_speed[max(0, i - 1):i + 2]
                lane_leaders_pos = lane_leaders_pos[max(0, i - 1):i + 2]
                lane_followers_pos = lane_followers_pos[max(0, i - 1):i + 2]
                if i == 0:
                    f_sp = [speed / max_speed
                            for speed in lane_followers_speed + rl_speed]
                    f_sp.insert(0, -1.)
                    l_sp = [speed / max_speed
                            for speed in lane_leaders_speed]
                    l_sp.insert(0, -1.)
                    f_pos = [-((self.k.vehicle.get_x_by_id(rl) - pos) % length / length)
                             for pos in lane_followers_pos]
                    f_pos.insert(0, -1.)
                    l_pos = [(pos - self.k.vehicle.get_x_by_id(rl)) % length / length
                             for pos in lane_leaders_pos]
                    l_pos.insert(0, -1.)

                elif i == max_lanes-1:
                    f_sp = [speed / max_speed
                            for speed in lane_followers_speed + rl_speed]
                    f_sp.insert(2, -1.)
                    l_sp = [speed / max_speed
                            for speed in lane_leaders_speed]
                    l_sp.insert(2, -1.)
                    f_pos = [-((self.k.vehicle.get_x_by_id(rl) - pos) % length / length)
                             for pos in
                             lane_leaders_pos]
                    f_pos.insert(2, -1.)
                    l_pos = [(pos - self.k.vehicle.get_x_by_id(rl)) % length / length
                             for pos in
                             lane_leaders_pos]
                    l_pos.insert(2, -1.)

                else:
                    f_sp = [speed / max_speed
                            for speed in lane_followers_speed + rl_speed]
                    l_sp = [speed / max_speed
                            for speed in lane_leaders_speed]
                    f_pos = [-((self.k.vehicle.get_x_by_id(rl) - pos) % length / length)
                             for pos in lane_leaders_pos]
                    l_pos = [(pos - self.k.vehicle.get_x_by_id(rl)) % length / length
                             for pos in lane_leaders_pos]

        positions = l_pos + f_pos
        speeds = l_sp + f_sp
        lanes = [self.k.vehicle.get_lane(rl) / (max_lanes - 1)]
        # lanes = [lane / max_lanes
        #          for lane in follower_lanes + leader_lanes + [self.k.vehicle.get_lane(rl)]]

        # coef = [rl_des, uns4IDM_p]

        # observation = np.array(speeds + positions + lanes + coef)
        observation = np.array(speeds + positions + lanes)
        return observation

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        for veh_id in self.visible:
            self.k.vehicle.set_observed(veh_id)