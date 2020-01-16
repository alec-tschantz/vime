# pylint: disable=assignment-from-no-return
import numpy as np
import logger


def train_model(model,
                replay_pool,
                n_train_batches=20,
                train_batch_size=10):
    """train dynamics model from replay pool"""
    obs_mean, obs_std, act_mean, act_std = replay_pool.mean_obs_act()

    _inputs = []
    _targets = []
    for _ in range(n_train_batches):
        batch = replay_pool.random_batch(train_batch_size)

        obs = (batch['observations'] - obs_mean) / \
            (obs_std + 1e-8)
        next_obs = (
            batch['next_observations'] - obs_mean) / (obs_std + 1e-8)
        act = (batch['actions'] - act_mean) / \
            (act_std + 1e-8)

        inputs = np.hstack(
            [obs, act])
        targets = next_obs

        _inputs.append(inputs)
        _targets.append(targets)

    for inputs, targets in zip(_inputs, _targets):
        model.train_fn(inputs, targets)

    test_model_accuracy(_inputs, _targets, model)


def test_model_accuracy(inputs, targets, model):
    """test accuracy of dynamics model"""
    acc = 0.
    for _inputs, _targets in zip(inputs, targets):
        _out = model.pred_fn(_inputs)
        _out = _out.numpy()
        acc += np.mean(np.square(_out - _targets))
    acc /= len(inputs)
    logger.log("Average Model Accuracy [{}]".format(acc))
    logger.log_value("model_accuracy", acc)


def eval_inf_gain(path,
                  model,
                  replay_pool,
                  inf_gain_itr=10,
                  inf_gain_batch_size=10):
    """evaluate information gain for given path"""
    obs_mean, obs_std, act_mean, act_std = replay_pool.mean_obs_act()
    obs = (path['observations'] - obs_mean) / (obs_std + 1e-8)
    act = (path['actions'] - act_mean) / (act_std + 1e-8)
    reward = path['rewards']

    obs_nxt = np.vstack([obs[1:]])
    _act = act[:-1].reshape(-1, 1)
    _inputs = np.hstack([obs[:-1], _act])
    _targets = obs_nxt

    _inf_gain = np.zeros(reward.shape)
    for j in range(int(np.ceil(obs.shape[0] / float(inf_gain_batch_size)))):
        model.save_old_params()

        start = j * inf_gain_batch_size
        end = np.minimum(
            (j + 1) * inf_gain_batch_size, obs.shape[0] - 1)

        for _ in range(inf_gain_itr):
            model.train_update_fn(_inputs[start:end], _targets[start:end])

        inf_gain = np.clip(float(model.kl_div_new_old()), 0, 1000)

        for k in np.arange(start, end):
            _inf_gain[k] = inf_gain

        model.reset_to_old_params()

    _inf_gain[-1] = _inf_gain[-2]
    return _inf_gain
