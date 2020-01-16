# pylint: disable=assignment-from-no-return
# pylint: disable=no-member

import numpy as np
import train_dynamics
import torch


def sample_paths(env, policy, model, replay_pool, itr,
                 n_paths=1, max_path_length=200, action_repeat=1,
                 calc_inf_gain=True, inf_gain_itr=10, inf_gain_batch_size=10):
    """sample a given number of paths"""
    paths = []
    for _ in range(n_paths):
        path = collect_one_path(env, policy, model, replay_pool, itr,
                                max_path_length=max_path_length, action_repeat=action_repeat,
                                calc_inf_gain=calc_inf_gain, inf_gain_itr=inf_gain_itr,
                                inf_gain_batch_size=inf_gain_batch_size)
        paths.append(path)
    return paths


def collect_one_path(env, policy, model, replay_pool, itr,
                     max_path_length, action_repeat,
                     calc_inf_gain, inf_gain_itr, inf_gain_batch_size):
    """collect single path and evaluate inf gain"""
    path = rollout(env, policy, max_path_length, action_repeat)
    path['raw_rewards'] = path['rewards']

    if itr > 0 and calc_inf_gain:
        inf_gain = train_dynamics.eval_inf_gain(path, model, replay_pool,
                                                inf_gain_itr=inf_gain_itr, inf_gain_batch_size=inf_gain_batch_size)
        path['inf_gain'] = inf_gain

    return path


def rollout(env, policy, max_path_length=np.inf, action_repeat=1):
    observations = []
    actions = []
    log_probs = torch.Tensor([])
    rewards = []

    o = env.reset()
    path_length = 0
    while path_length < max_path_length:
        a, log_prob = policy.get_action(o)
        for _ in range(action_repeat):
            next_o, r, d, _ = env.step(a)
            observations.append(o)
            rewards.append(r)
            actions.append(a)
            log_probs = torch.cat([log_probs, log_prob.reshape(1)])
            path_length += 1
            if d:
                break
            o = next_o
        if d:
            break

    return dict(observations=np.array(observations),
                actions=np.array(actions),
                rewards=np.array(rewards),
                log_probs=log_probs,
                path_length=path_length)
