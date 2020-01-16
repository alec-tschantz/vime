# pylint: disable=no-member
import torch
import numpy as np
import logger

def add_paths_to_pool(pool, data):
    for path in data['paths']:
        path_len = len(path['rewards'])
        for i in range(path_len):
            obs = path['observations'][i]
            act = path['actions'][i]
            rew = path['rewards'][i]
            term = (i == path_len - 1)
            pool.add_sample(obs, act, rew, term)


def process_samples(itr, paths, calc_inf_gain=True, inf_factor=1):
    if itr > 0 and calc_inf_gain:
        inf_gains = []
        for i in range(len(paths)):
            inf_gains.append(paths[i]['inf_gain'])

        inf_gains_flat = np.hstack(inf_gains)
        avg_inf_gain = np.round(np.mean(inf_gains_flat), 5)
        logger.log("Average Information Gain [{}]".format(avg_inf_gain))
        logger.log_value("inf_gain", avg_inf_gain)

        for i in range(len(paths)):
            paths[i]['rewards'] = paths[i]['rewards'] + inf_factor * inf_gains_flat[i]

    observations = np.concatenate(
        [path["observations"] for path in paths], axis=0)
    actions = np.concatenate([path["actions"]
                              for path in paths], axis=0)
    rewards = np.concatenate([path['rewards']
                              for path in paths], axis=0)
    raw_rewards = np.concatenate(
        [path['raw_rewards'] for path in paths], axis=0)
    log_probs = torch.cat([path['log_probs'] for path in paths])

    mean_reward = np.sum(raw_rewards) / len(paths)
    logger.log("Mean Reward [{}]".format(mean_reward))
    logger.log_value("mean_reward", mean_reward)

    return dict(observations=observations,
                actions=actions,
                rewards=rewards,
                log_probs=log_probs,
                raw_rewards=raw_rewards,
                paths=paths)

