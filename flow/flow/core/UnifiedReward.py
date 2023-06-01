"""A series of reward functions."""

from flow.core.params import SumoParams
from gym.spaces import Box, Tuple
import numpy as np

from collections import defaultdict
from functools import reduce

from requests import head

def total_lc_reward(env, rl_action):
    reward_dict = {
        'rl_desired_speed': rl_desired_speed(env),
        'rl_action_penalty': rl_action_penalty(env, rl_action),
        'uns4IDM_penalty': unsafe_distance_penalty4IDM(env),
        'meaningless_penalty': meaningless_penalty(env),
        'target_velocity': target_velocity(env),
    }
    return reward_dict

def target_velocity(env):

    return 0

def rl_desired_speed(env, rl):
    vel = np.array(env.k.vehicle.get_speed(rl))
    rl_des = env.initial_config.reward_params.get('rl_desired_speed', 0)
    target_vel = env.initial_config.reward_params.get('target_velocity', 0)
    regulation_speed = env.initial_config.reward_params.get('regulation_velocity', 0)

    if rl_des == 0:
        return 0

    if vel < -100:
        return 0.

    if vel <= target_vel:
        cost = vel
        rwd = cost/target_vel
    else:
        cost = regulation_speed-vel
        rwd = cost/(regulation_speed-target_vel)

    return rwd

def unsafe_distance_penalty4IDM(env, rl, tail_way, tail_speed):
    uns4IDM_p = env.initial_config.reward_params.get('uns4IDM_penalty', 0)
    T = 1
    a = 1
    b = 1
    s0 = 2
    v = env.k.vehicle.get_speed(rl)
    tw = tail_way
    rwd = 0

    if env.k.vehicle.get_last_lc(rl) == env.time_counter:
        if abs(tw) < 1e-3:
            tw = 1e-3
        else:
            follow_vel = tail_speed
            s_star = s0 + max(
                0, follow_vel * T + follow_vel * (follow_vel - v) /
                (2 * np.sqrt(a * b)))
        rwd = uns4IDM_p * max(-5, min(0, 1 - (s_star / tw) ** 2))

    return rwd

def rl_action_penalty(env, actions, rl):
    action_penalty = env.initial_config.reward_params.get('rl_action_penalty', 0)

    if actions is None or action_penalty == 0:
        return 0

    actions = actions[rl]

    #  boolean condition
    if len(actions) == 2:
        direction = actions[1::2]
        for i in range(len(direction)):
            if direction[i] <= -0.333:
                direction[i] = -1
            elif direction[i] >= 0.333:
                direction[i] = 1
            else:
                direction[i] = 0

    elif len(actions) == 4:
        direction = actions[1:]

        if direction[0] == 1:
            direction = np.array([-1])
        elif direction[1] == 1:
            direction = np.array([0])
        else:
            direction = np.array([1])

    reward = 0
    if direction:
        if env.k.vehicle.get_previous_lane(rl) == env.k.vehicle.get_lane(rl):
            reward -= action_penalty

    return reward


def meaningless_penalty(env, rl, prev_headway, headway):
    mlp = env.initial_config.reward_params.get('meaningless_penalty', 0)
    reward = 0
    lenght = env.k.network.length()
    lc_pen = 5

    if mlp:
        if env.k.vehicle.get_last_lc(rl) == env.time_counter:
            headway_criterion = headway-prev_headway
            semi_reward = -(headway_criterion-lc_pen)
            reward -= mlp*(semi_reward/lenght)

    return reward
