"""A series of reward functions."""

from gym.spaces import Box, Tuple
import numpy as np

from collections import defaultdict
from functools import reduce

def total_lc_reward(env, rl_action):
    """Put all of the reward functions we consider into list

    Parameters
    ----------
    env :
        the environment variable, which contains information on the current
        state of the system.

    rl_action : gym.spaces.Space

    Returns
    -------
    Dict
        Dictionary of rewards
    """
    reward_dict = {
        'rl_desired_speed': rl_desired_speed(env),
        'simple_lc_penalty': simple_lc_penalty(env),
        'dc3_penalty': follower_decel_penalty(env),
        'unsafe_penalty': unsafe_distance_penalty(env),
        'rl_action_penalty': rl_action_penalty(env, rl_action),
        'acc_penalty': punish_accelerations(env, rl_action),
        'uns4IDM_penalty': unsafe_distance_penalty4IDM(env),
        'meaningless_penalty': meaningless_penalty(env),
    }
    return reward_dict

def rl_desired_speed(env):
    vel = np.array(env.k.vehicle.get_speed(env.k.vehicle.get_rl_ids()))
    rl_des = env.initial_config.reward_params.get('rl_desired_speed', 0)

    if rl_des == 0:
        return 0

    if any(vel < -100):
        return 0.
    if len(vel) == 0:
        return 0.

    rls = env.k.vehicle.get_rl_ids()

    vel = np.array(env.k.vehicle.get_speed(rls))
    num_vehicles = len(rls)

    target_vel = env.env_params.additional_params['target_velocity']
    max_cost = np.array([target_vel] * num_vehicles)
    max_cost = np.linalg.norm(max_cost)

    cost = vel - target_vel
    cost = np.linalg.norm(cost)

    # epsilon term (to deal with ZeroDivisionError exceptions)
    eps = np.finfo(np.float32).eps

    return rl_des * (1 - (cost / (max_cost + eps)))

def unsafe_distance_penalty4IDM(env):
    uns4IDM_p = env.initial_config.reward_params.get('uns4IDM_penalty', 0)
    rls = env.k.vehicle.get_rl_ids()

    #  Parameter of IDM
    T = 1
    a = 1
    b = 1.5
    s0 = 2

    v = env.k.vehicle.get_speed(rls)[0]
    # rl_lane = env.k.vehicle.get_lane(rls)
    tw = env.k.vehicle.get_tailway(rls)[0]

    follow_id = env.k.vehicle.get_follower(rls)[0]
    # if rl_lane == [0]:
    #     tw = env.k.vehicle.get_tailway(rls)[0]
    #     follow_id = env.k.vehicle.get_follower(rls)[0]
    # elif rl_lane == [1]:
    #     tw = env.k.vehicle.get_tailway(rls)[1]
    #     follow_id = env.k.vehicle.get_follower(rls)[1]

    if abs(tw) < 1e-3:
        tw = 1e-3

    if follow_id is None or follow_id == '':
        s_star = 0

    else:
        follow_vel = env.k.vehicle.get_speed(follow_id)
        s_star = s0 + max(
            0, follow_vel * T + follow_vel * (v - follow_vel) /
            (2 * np.sqrt(a * b)))

    rwd = uns4IDM_p * max(-5, min(0, 1 - (s_star / tw) ** 2))
    # if rwd < - 0:
        # print('rwd:{}, tw:{}\n'.format(rwd, tw))
    return rwd

def punish_accelerations(env,rl_action):
    acc_p = env.initial_config.reward_params.get('acc_penalty', 0)

    if rl_action is None:
        return 0
    else:
        acc = rl_action[0:1]
        mean_actions = np.mean(np.abs(np.array(acc)))
        accel_threshold = 0

        return acc_p * (accel_threshold - mean_actions)

def rl_action_penalty(env, rl_action):
    """

    Parameters
    ----------
    env: flow.envs.ring.my_lane_change_accel.MyLaneChangeAccelEnv
    rl_action

    Returns
    -------

    """
    def new_softmax(a):
        c = np.max(a)  # 최댓값
        exp_a = np.exp(a - c)  # 각각의 원소에 최댓값을 뺀 값에 exp를 취한다. (이를 통해 overflow 방지)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    action_penalty = env.initial_config.reward_params.get('rl_action_penalty', 0)
    if rl_action is None or action_penalty == 0:
        return 0

    rls = env.k.vehicle.get_rl_ids()
    #  boolean condition
    lc_failed = np.array(env.last_lane) == np.array(env.k.vehicle.get_lane(rls))
    lc_rl_action = np.zeros_like(rl_action)

    if isinstance(env.action_space, Box):
        lc_rl_action = np.array(rl_action[1:])
        d_list = new_softmax(lc_rl_action)
        for i in range(len(d_list)):
            if d_list[i] == max(d_list):
                d_list[i] = 1.
            else:
                d_list[i] = 0.

        if d_list[0] == 1:
            lc_rl_action = np.array([-1.])
        elif d_list[1] == 1:
            lc_rl_action = np.array([0.])
        elif d_list[2] == 1:
            lc_rl_action = np.array([1.])
    # elif isinstance(env.action_space, Tuple):
    #     lc_rl_action = rl_action[1] - 1
    if any(lc_rl_action) and any(lc_failed):
        return -action_penalty * sum(lc_failed)
    else:
        return 0

# def meaningless_penalty(env):
#     mlp = env.initial_config.reward_params.get('meaningless_penalty', 0)
#     reward = 0
#     if mlp:
#         for veh_id in env.k.vehicle.get_rl_ids():
#             if env.k.vehicle.get_last_lc(veh_id) == env.time_counter:
#                 lane_leaders = env.k.vehicle.get_lane_leaders(veh_id)
#                 headway = [env.k.vehicle.get_x_by_id(leader) for leader in lane_leaders]
#                 lane_headway = (headway[env.k.vehicle.get_lane(veh_id)] -
#                                 env.k.vehicle.get_x_by_id(veh_id))% env.k.network.length()
#                 if lane_headway > 15:
#                     reward -= mlp
#                 else:
#                     reward += mlp
#
#     return reward

def meaningless_penalty(env):
    mlp = env.initial_config.reward_params.get('meaningless_penalty', 0)
    reward = 0

    if mlp:
        for veh_id in env.k.vehicle.get_rl_ids():
            # print(env.time_counter, env.k.vehicle.get_lane(veh_id))
            if env.k.vehicle.get_last_lc(veh_id) == env.time_counter:
                lane_leaders = env.k.vehicle.get_lane_leaders(veh_id)
                headway = [(env.k.vehicle.get_x_by_id(leader) - env.k.vehicle.get_x_by_id(veh_id))
                           % env.k.network.length() / env.k.network.length() for leader in lane_leaders]

                # lane_headway = [(headway[env.k.vehicle.get_lane(0)] -
                #                 env.k.vehicle.get_x_by_id(veh_id))% env.k.network.length() ,
                #                 (headway[env.k.vehicle.get_lane(1)] -
                #                 env.k.vehicle.get_x_by_id(veh_id)) % env.k.network.length()]
                # print('headway:{},RL_lane:{}, lane_leader:{}'.format(headway, env.k.vehicle.get_lane(veh_id),lane_leaders))
                if env.k.vehicle.get_lane(veh_id) == 0 and headway[0] < headway[1]:
                    reward -= mlp * (headway[1])
                    # print('AT lane0 {},{},{}'.format(headway, reward, env.time_counter))
                elif env.k.vehicle.get_lane(veh_id) == 1 and headway[1] < headway[0]:
                    reward -= mlp * (headway[0])
                    # print('AT lane1 {},{},{}'.format(headway, reward, env.time_counter))
                # print('time:{}, rl_lane:{}, reward:{}, headway:{}'.format(env.time_counter, env.k.vehicle.get_lane(veh_id), reward, headway))

    return reward

def simple_lc_penalty(env):
    sim_lc_penalty = env.initial_config.reward_params.get('simple_lc_penalty', 0)
    if not sim_lc_penalty:
        return 0
    reward = 0
    for veh_id in env.k.vehicle.get_rl_ids():
        if env.k.vehicle.get_last_lc(veh_id) == env.time_counter:
            reward -= sim_lc_penalty
    print(reward)
    return reward

def follower_decel_penalty(env):
    """Reward function used to reward the RL vehicles cause the emergency stop of non RL vehicles

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.

    Returns
    -------
    float
        reward value
    """
    dc3_p = env.initial_config.reward_params.get('dc3_penalty', 0)
    if not dc3_p:
        return 0
    reward = 0
    threshold = -0.2
    rls = env.k.vehicle.get_rl_ids()
    max_decel = env.env_params.additional_params['max_decel']
    for rl in rls:
        follower = env.k.vehicle.get_follower(rl)
        if follower is not None:
            accel = env.k.vehicle.get_accel(env.k.vehicle.get_follower(rl)) or 0
            if accel < threshold:
                f = lambda x: x if x < 1 else np.log(x) + 1
                pen = dc3_p * f(abs(2 * accel / max_decel))
                reward -= pen

    return reward

def unsafe_distance_penalty(env):
    unsafe_p = env.initial_config.reward_params.get('unsafe_penalty', 0)
    rls = env.k.vehicle.get_rl_ids()
    reward = 0
    for rl in rls:
        follower = env.k.vehicle.get_follower(rl)
        if follower is not None and unsafe_p:
            tailway = env.k.vehicle.get_tailway(rl)
            gap = 5 + env.k.vehicle.get_speed(env.k.vehicle.get_follower(rl)) ** 2 / (
                        2 * env.env_params.additional_params['max_decel'])
            if tailway < gap:
                pen = unsafe_p * (gap - tailway) / gap
                reward -= pen
    return reward

