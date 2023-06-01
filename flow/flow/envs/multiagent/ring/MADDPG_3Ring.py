import numpy as np
from gym.spaces.box import Box
import random
from scipy.optimize import fsolve
from copy import deepcopy

from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.envs.multiagent.base import MultiEnv
from flow.envs.ring.wave_attenuation import v_eq_max_function


ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration of autonomous vehicles
    'max_accel': 1,
    # maximum deceleration of autonomous vehicles
    'max_decel': 1,
    # bounds on the ranges of ring road lengths the autonomous vehicle is
    # trained on
    'ring_length': [220, 270],
}


class MADDPGForwardLaneChange3RingEnv(MultiEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def observation_space(self):
        """See class definition."""
        pass

    @property
    def action_space(self):
        max_decel = self.env_params.additional_params["max_decel"]
        max_accel = self.env_params.additional_params["max_accel"]

        lb = [-abs(max_decel), -1]
        ub = [max_accel, 1.]
        shape = self.initial_vehicles.num_rl_vehicles,
        return Box(np.array(lb), np.array(ub), dtype=np.float32)

    def get_state(self):
        """See class definition."""
        pass

    def _apply_rl_actions(self, rl_actions):
        """Split the accelerations by ring."""
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

    def Coef(self):
        global coef_log
        np.set_printoptions(precision=4)
        rls = self.k.vehicle.get_rl_ids()

        if self.time_counter == 0:
            coef_log = []
            des = list(np.random.choice([i for i in np.arange(1.5, 2.1, 0.1)], len(rls), replace=True))
            uns = list(np.random.choice([i for i in np.arange(0.0, 0.6, 0.1)], len(rls), replace=True))
            mlp = list(np.random.choice([i for i in np.arange(0.0, 0.6, 0.1)], len(rls), replace=True))

            des_dict = {rls[i]: des[i] for i in range(len(rls))}
            uns_dict = {rls[i]: uns[i] for i in range(len(rls))}
            mlp_dict = {rls[i]: mlp[i] for i in range(len(rls))}
            coef_log.append([des_dict, uns_dict, mlp_dict])

        else:
            des_dict = coef_log[0][0]
            uns_dict = coef_log[0][1]
            mlp_dict = coef_log[0][2]

        return des_dict, uns_dict, mlp_dict

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        global r, pen1, pen2, pen3
        if rl_actions is None:
            return {}

        rls = self.k.vehicle.get_rl_ids()

        reward = {}  # reward = 0

        # Training
        rl_des, uns4IDM_p, mlp = self.Coef()
        # print('compute_reward:{},{},{}'.format(rl_des, uns4IDM_p, mlp))

        # #  Execution
        # rl_des = {}
        # uns4IDM_p = {}
        # mlp = {}
        # tv = {}
        # for k in range(len(rls)):
        #     rl_des[rls[k]] = self.initial_config.reward_params.get('rl_desired_speed', 0)[k]
        #     uns4IDM_p[rls[k]] = self.initial_config.reward_params.get('uns4IDM_penalty', 0)[k]
        #     mlp[rls[k]] = self.initial_config.reward_params.get('meaningless_penalty', 0)[k]
        #     tv[rls[k]] = self.initial_config.reward_params.get('target_velocity', 0)[k]

        rl_action_p = self.initial_config.reward_params.get('rl_action_penalty', 0)

        for rl in rls:
            r = 0
            if rl_des:
                if self.k.vehicle.get_speed(rl) > 0.:
                    vel = np.array([
                        self.k.vehicle.get_speed(rl)
                    ])

                    # num_vehicles = len(rls)

                    if any(vel < -100) or kwargs['fail']:
                        return 0.

                    target_vel = self.env_params.additional_params['target_velocity']
                    max_cost = np.array([target_vel])
                    max_cost = np.linalg.norm(max_cost)

                    cost = vel - target_vel
                    cost = np.linalg.norm(cost)

                    # epsilon term (to deal with ZeroDivisionError exceptions)
                    eps = np.finfo(np.float32).eps

                    # r = rl_des * (1 - (cost / (max_cost + eps)))
                    r = rl_des[rl] * (1 - (cost / (max_cost + eps)))

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
                            # pen1 -= mlp * (headway[self.k.vehicle.get_previous_lane(veh_id)]
                            #                - headway[self.k.vehicle.get_lane(veh_id)]) * 130 \
                            #         * (headway[self.k.vehicle.get_previous_lane(veh_id)])
                            pen1 -= mlp[veh_id] * (headway[self.k.vehicle.get_previous_lane(veh_id)]
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

                    # pen2 = uns4IDM_p * max(-5, min(0, 1 - (s_star / tw) ** 2))
                    pen2 = uns4IDM_p[veh_id] * max(-5, min(0, 1 - (s_star / tw) ** 2))

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

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        for rl_id in self.k.vehicle.get_rl_ids():
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id
            self.k.vehicle.set_observed(lead_id)


class MADDPGForwardLaneChange3RingPOEnv(MADDPGForwardLaneChange3RingEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        self.num_lanes = max(self.k.network.num_lanes(edge)
                             for edge in self.k.network.get_edge_list())
        self.visible = []

    @property
    def observation_space(self):
        return Box(
            low=-1,
            high=1,
            shape=(17, ),
            dtype=np.float32)

    def get_state(self):
        """See class definition."""
        # normalizers
        max_speed = self.k.network.max_speed()
        # print(max_speed)
        length = self.k.network.length()
        max_lanes = max(
            self.k.network.num_lanes(edge)
            for edge in self.k.network.get_edge_list())

        # #  Execution
        # rls = self.k.vehicle.get_rl_ids()
        # RD = {}
        # UP = {}
        # MLP = {}
        # TV = {}
        #
        # for k in range(len(rls)):
        #     RD[rls[k]] = self.initial_config.reward_params.get('rl_desired_speed', 0)[k]
        #     UP[rls[k]] = self.initial_config.reward_params.get('uns4IDM_penalty', 0)[k]
        #     MLP[rls[k]] = self.initial_config.reward_params.get('meaningless_penalty', 0)[k]
        #     TV[rls[k]] = self.initial_config.reward_params.get('target_velocity', 0)[k]

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

            # Sampling Characteristic (Training)
            rl_des, uns4IDM_p, mlp = self.Coef()

            # # Execution
            # rl_des = RD[rl_id]
            # uns4IDM_p = UP[rl_id]
            # mlp = MLP[rl_id]
            # tv = TV[rl_id]

            for i in range(0, max_lanes):

                if self.k.vehicle.get_lane(rl_id) == i:
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
                        f_pos = [((self.k.vehicle.get_x_by_id(rl_id) - pos) % length / length)
                                 for pos in lane_followers_pos]
                        f_pos.insert(0, -1.)
                        l_pos = [(pos - self.k.vehicle.get_x_by_id(rl_id)) % length / length
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
                        f_sp = [(speed - rl_speed) / max_speed
                                for speed in lane_followers_speed]
                        l_sp = [(speed - rl_speed) / max_speed
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
                            positions[i - 1] = -1.0

                        else:
                            pass

                    coef = [i / 3 for i in [rl_des[rl_id], uns4IDM_p[rl_id], mlp[rl_id]]]
                    observation = np.array(speeds + positions + lanes + coef)
                    obs.update({rl_id: observation})

        return obs

    # def reset(self, new_inflow_rate=None):
    #     """See parent class.
    #
    #     The sumo instance is reset with a new ring length, and a number of
    #     steps are performed with the rl vehicle acting as a human vehicle.
    #     """
    #     # skip if ring length is None
    #     if self.env_params.additional_params['ring_length'] is None:
    #         return super().reset()
    #
    #     # reset the step counter
    #     self.step_counter = 0
    #
    #     # update the network
    #     initial_config = InitialConfig(bunching=50, min_gap=0)
    #     length = random.randint(
    #         self.env_params.additional_params['ring_length'][0],
    #         self.env_params.additional_params['ring_length'][1])
    #     additional_net_params = {
    #         'length':
    #             length,
    #         'lanes':
    #             self.net_params.additional_params['lanes'],
    #         'speed_limit':
    #             self.net_params.additional_params['speed_limit'],
    #         'resolution':
    #             self.net_params.additional_params['resolution']
    #     }
    #     net_params = NetParams(additional_params=additional_net_params)
    #
    #     self.network = self.network.__class__(
    #         self.network.orig_name, self.network.vehicles,
    #         net_params, initial_config)
    #     self.k.vehicle = deepcopy(self.initial_vehicles)
    #     self.k.vehicle.kernel_api = self.k.kernel_api
    #     self.k.vehicle.master_kernel = self.k
    #
    #     # solve for the velocity upper bound of the ring
    #     v_guess = 4
    #     v_eq_max = fsolve(v_eq_max_function, np.array(v_guess),
    #                       args=(len(self.initial_ids), length))[0]
    #
    #     print('\n-----------------------')
    #     print('ring length:', net_params.additional_params['length'])
    #     print('v_max:', v_eq_max)
    #     print('-----------------------')
    #
    #     # restart the sumo instance
    #     self.restart_simulation(
    #         sim_params=self.sim_params,
    #         render=self.sim_params.render)
    #
    #     # perform the generic reset function
    #     return super().reset()
