from flow.envs.ring.accel import AccelEnv
from flow.envs.ring.lane_change_accel import LaneChangeAccelEnv
from flow.core import lane_change_rewards as rewards
from flow.core.params import InitialConfig

from gym.spaces.box import Box
from gym.spaces.tuple import Tuple
from gym.spaces.multi_discrete import MultiDiscrete

from sympy import *
import numpy as np
import random

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


class TD3LCIStoPoissonAccelEnv(AccelEnv):
    """Fully observable lane change and acceleration environment.

    This environment is used to train autonomous vehicles to improve traffic
    flows when lane-change and acceleration actions are permitted by the rl
    agent.

    """
    coef_log = []

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

    def Coef(self):
        global coef_log
        global c_log
        np.set_printoptions(precision=4)

        if self.time_counter == 0:
            coef_log = []
            c_log = []

            rl_des_lambda = random.sample([i for i in np.arange(0.3, 0.8, 0.1)], 1)[-1]
            uns4IDM_p_lambda = random.sample([i for i in np.arange(0.3, 0.8, 0.1)], 1)[-1]
            mlp_lambda = random.sample([i for i in np.arange(0.3, 0.8, 0.1)], 1)[-1]


            coef_log.append([rl_des_lambda, uns4IDM_p_lambda, mlp_lambda])

            C1 = self.poisson_dist(rl_des_lambda * 10, 0, 20)
            C2 = self.poisson_dist(uns4IDM_p_lambda * 10, 0, 20)
            C3 = self.poisson_dist(mlp_lambda * 10, 0, 20)
            c_log.append([C1, C2, C3])

        else:
            rl_des_lambda = coef_log[0][0]
            uns4IDM_p_lambda = coef_log[0][1]
            mlp_lambda = coef_log[0][2]

            C1 = c_log[0][0]
            C2 = c_log[0][1]
            C3 = c_log[0][2]

        return C1, C2, C3, rl_des_lambda, uns4IDM_p_lambda, mlp_lambda

    def execution_coef(self, C1_lambda, C2_lambda, C3_lambda):
        global coef_log
        global c_log
        np.set_printoptions(precision=4)

        if self.time_counter == 0:
            coef_log = []
            c_log = []

            rl_des_lambda = C1_lambda
            uns4IDM_p_lambda = C2_lambda
            mlp_lambda = C3_lambda

            coef_log.append([rl_des_lambda, uns4IDM_p_lambda, mlp_lambda])

            C1 = self.poisson_dist(rl_des_lambda * 10, 0, 20)
            C2 = self.poisson_dist(uns4IDM_p_lambda * 10, 0, 20)
            C3 = self.poisson_dist(mlp_lambda * 10, 0, 20)

            c_log.append([C1, C2, C3])

        else:
            rl_des_lambda = coef_log[0][0]
            uns4IDM_p_lambda = coef_log[0][1]
            mlp_lambda = coef_log[0][2]
            C1 = c_log[0][0]
            C2 = c_log[0][1]
            C3 = c_log[0][2]

        return C1, C2, C3, rl_des_lambda, uns4IDM_p_lambda, mlp_lambda

    def poisson_dist(self, lamb, a, b):
        K = Symbol('K')
        PMF = K * (((lamb ** K) * (np.exp(1) ** (-lamb))) / factorial(K))
        exp = Integral(PMF, (K, a, b)).doit().evalf()
        return 0.1 * float(exp)

    def compute_reward(self, rl_actions, **kwargs):
        rls = self.k.vehicle.get_rl_ids()
        reward = 0

        # # Training
        # C1, C2, C3, _, _, _ = self.Coef()
        # # print('compute_reward:{},{},{}'.format(rl_des, uns4IDM_p, mlp))

        #  Execution
        rl_des = self.initial_config.reward_params.get('rl_desired_speed', 0)
        uns4IDM_p = self.initial_config.reward_params.get('uns4IDM_penalty', 0)
        mlp = self.initial_config.reward_params.get('meaningless_penalty', 0)

        C1, C2, C3, _, _, _ = self.execution_coef(rl_des, uns4IDM_p, mlp)

        rl_action_p = self.initial_config.reward_params.get('rl_action_penalty', 0)

        rwds = defaultdict(int)

        for rl in rls:
            if C1:
                if self.k.vehicle.get_speed(rl) > 0.:
                    vel = np.array(self.k.vehicle.get_speed(self.k.vehicle.get_rl_ids()))

                    if C1 == 0:
                        return 0

                    if any(vel < -100):
                        return 0.
                    if len(vel) == 0:
                        return 0.

                    vel = np.array(self.k.vehicle.get_speed(rls))
                    num_vehicles = len(rls)

                    target_vel = self.env_params.additional_params['target_velocity']
                    max_cost = np.array([target_vel] * num_vehicles)
                    max_cost = np.linalg.norm(max_cost)

                    cost = vel - target_vel
                    cost = np.linalg.norm(cost)

                    # epsilon term (to deal with ZeroDivisionError exceptions)
                    eps = np.finfo(np.float32).eps

                    C1 = C1 + 1
                    r = C1 * (1 - (cost / (max_cost + eps)))
                    reward += r
                    rwds['rl_desired_speed'] += r

            follower = self.k.vehicle.get_follower(rl)
            leader = self.k.vehicle.get_leader(rl)

            if leader is not None:
                if C2:
                    pen = 0
                    if C2:
                        for veh_id in self.k.vehicle.get_rl_ids():
                            if self.k.vehicle.get_last_lc(veh_id) == self.time_counter:
                                lane_leaders = self.k.vehicle.get_lane_leaders(veh_id)
                                headway = [(self.k.vehicle.get_x_by_id(leader) - self.k.vehicle.get_x_by_id(veh_id))
                                           % self.k.network.length() / self.k.network.length() for leader in
                                           lane_leaders]
                                # FOR N LANE
                                if headway[self.k.vehicle.get_previous_lane(veh_id)] - headway[
                                    self.k.vehicle.get_lane(veh_id)] > 5:
                                    pen -= C2 * (headway[self.k.vehicle.get_previous_lane(veh_id)])

                    reward += pen
                    rwds['meaningless_penalty'] += pen

            if follower is not None:
                if C3:
                    T = 1
                    a = 1
                    b = 1
                    s0 = 2

                    v = self.k.vehicle.get_speed(rls)[0]
                    tw = self.k.vehicle.get_tailway(rls)[0]
                    follow_id = self.k.vehicle.get_follower(rls)[0]

                    if abs(tw) < 1e-3:
                        tw = 1e-3

                    if follow_id is None or follow_id == '':
                        s_star = 0

                    else:
                        follow_vel = self.k.vehicle.get_speed(follow_id)
                        s_star = s0 + max(
                            0, follow_vel * T + follow_vel * (follow_vel - v) /
                               (2 * np.sqrt(a * b)))

                    pen = C3 * max(-5, min(0, 1 - (s_star / tw) ** 2))
                    reward += pen
                    rwds['uns4IDM_penalty'] += pen

            if rl_action_p:
                pen = rewards.rl_action_penalty(self, rl_actions)
                reward += pen
                rwds['rl_action_penalty'] += pen

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

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        if self.k.vehicle.num_rl_vehicles > 0:
            for veh_id in self.k.vehicle.get_human_ids():
                self.k.vehicle.set_observed(veh_id)


class TD3LCIStoPoissonAccelPOEnv(TD3LCIStoPoissonAccelEnv):

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
            shape=(3 * 2 * 2 + 2 + 3,),
            dtype=np.float32)

    def get_state(self):
        """See class definition."""
        # normalizers
        max_speed = self.k.network.max_speed()
        length = self.k.network.length()
        max_lanes = max(
            self.k.network.num_lanes(edge)
            for edge in self.k.network.get_edge_list())

        # # # Training random coef
        # _, _, _, rl_des_mu, uns4IDM_p_mu, mlp_mu = self.Coef()
        # # # print('get_state:{},{},{}'.format(rl_des, uns4IDM_p, mlp))

        #  Execution
        rl_des = self.initial_config.reward_params.get('rl_desired_speed', 0)
        uns4IDM_p = self.initial_config.reward_params.get('uns4IDM_penalty', 0)
        mlp = self.initial_config.reward_params.get('meaningless_penalty', 0)

        _, _, _, rl_des_mu, uns4IDM_p_mu, mlp_mu = self.execution_coef(rl_des, uns4IDM_p, mlp)

        # NOTE: this works for only single agent environmnet
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

        for i in range(0, max_lanes):
            # print(max_lanes)
            if self.k.vehicle.get_lane(rl) == i:
                lane_followers_speed = lane_followers_speed[max(0, i - 1):i + 2]
                lane_leaders_speed = lane_leaders_speed[max(0, i - 1):i + 2]
                lane_leaders_pos = lane_leaders_pos[max(0, i - 1):i + 2]
                lane_followers_pos = lane_followers_pos[max(0, i - 1):i + 2]

                if i == 0:
                    f_sp = [(speed - rl_speed) / max_speed
                            for speed in lane_followers_speed]
                    f_sp.insert(0, -1.)
                    l_sp = [(speed - rl_speed) / max_speed
                            for speed in lane_leaders_speed]
                    l_sp.insert(0, -1.)
                    f_pos = [((self.k.vehicle.get_x_by_id(rl) - pos) % length / length)
                             for pos in lane_followers_pos]
                    f_pos.insert(0, -1.)
                    l_pos = [(pos - self.k.vehicle.get_x_by_id(rl)) % length / length
                             for pos in lane_leaders_pos]
                    l_pos.insert(0, -1.)
                    lanes = [0.]

                elif i == max_lanes - 1:
                    f_sp = [(speed - rl_speed) / max_speed
                            for speed in lane_followers_speed]
                    f_sp.insert(2, -1.)
                    l_sp = [(speed - rl_speed) / max_speed
                            for speed in lane_leaders_speed]
                    l_sp.insert(2, -1.)
                    f_pos = [((self.k.vehicle.get_x_by_id(rl) - pos) % length / length)
                             for pos in
                             lane_followers_pos]
                    f_pos.insert(2, -1.)
                    l_pos = [(pos - self.k.vehicle.get_x_by_id(rl)) % length / length
                             for pos in
                             lane_leaders_pos]
                    l_pos.insert(2, -1.)
                    lanes = [1.]

                else:
                    f_sp = [(speed - rl_speed) / max_speed
                            for speed in lane_followers_speed]
                    l_sp = [(speed - rl_speed) / max_speed
                            for speed in lane_leaders_speed]
                    f_pos = [((self.k.vehicle.get_x_by_id(rl) - pos) % length / length)
                             for pos in lane_followers_pos]
                    l_pos = [(pos - self.k.vehicle.get_x_by_id(rl)) % length / length
                             for pos in lane_leaders_pos]
                    lanes = [0.5]

                rl_sp = [rl_speed / max_speed]
                positions = l_pos + f_pos
                speeds = rl_sp + l_sp + f_sp

        # Coef scaling
        char_mu = [i for i in [rl_des_mu, uns4IDM_p_mu, mlp_mu]]
        observation = np.array(speeds + positions + lanes + char_mu)

        return observation

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        for veh_id in self.visible:
            self.k.vehicle.set_observed(veh_id)