# pylint: disable=no-member
import numpy as np
import torch
from tqdm import tqdm
import sampler
from replay_pool import ReplayPool
from policy import PolicyOptimizer
import train_dynamics
import utils
import logger


class Experiment(object):

    def __init__(self,
                 policy,
                 model,
                 env,
                 train_model=True,
                 calc_inf_gain=True,
                 exp_name="_exp",
                 start_itr=0,
                 n_itr=500,
                 n_paths=25,
                 max_path_length=200,
                 action_repeat=1,
                 n_train_batches=100,
                 train_batch_size=10,
                 inf_gain_itr=5,
                 inf_gain_batch_size=1,
                 inf_factor=1,
                 min_pool_size=200,
                 max_pool_size=100000):

        self.policy = policy
        self.model = model
        self.env = env
        self.policy_opt = PolicyOptimizer(self.policy)

        self.exp_name = exp_name
        self.calc_inf_gain = calc_inf_gain
        self.train_model = train_model

        self.start_itr = start_itr
        self.n_itr = n_itr
        self.n_paths = n_paths
        self.max_path_length = max_path_length
        self.action_repeat = action_repeat

        self.n_train_batches = n_train_batches
        self.train_batch_size = train_batch_size

        self.inf_gain_itr = inf_gain_itr
        self.inf_gain_batch_size = inf_gain_batch_size
        self.inf_factor = inf_factor

        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size

        self.obs_dim = np.prod(self.env.observation_space.shape)
        self.act_dim = np.prod(self.env.action_space.shape)

        self.pool = ReplayPool(
            self.max_pool_size, observation_shape=self.obs_dim, action_dim=self.act_dim)

        logger.set_log_file(self.exp_name)

    def train(self):
        for itr in tqdm(np.arange(self.start_itr, self.n_itr)):
            logger.log("----\nIteration {}\n----".format(itr))
            paths = self.obtain_samples(itr)
            data = self.process_samples(itr, paths)

            if self.train_model:
                self.add_paths_to_pool(data)
                if self.pool.size >= self.min_pool_size:

                    train_dynamics.train_model(self.model, self.pool,
                                               n_train_batches=self.n_train_batches, train_batch_size=self.train_batch_size)

            self.optimize_policy(itr, data)
            logger.print_log()
            logger.save_values()

    def obtain_samples(self, itr):
        paths = sampler.sample_paths(
            self.env, self.policy, self.model, self.pool, itr,
            n_paths=self.n_paths, max_path_length=self.max_path_length, action_repeat=self.action_repeat,
            calc_inf_gain=self.calc_inf_gain, inf_gain_itr=self.inf_gain_itr,
            inf_gain_batch_size=self.inf_gain_batch_size)
        return paths

    def process_samples(self, itr, paths):
        return utils.process_samples(itr, paths,
                                     calc_inf_gain=self.calc_inf_gain, inf_factor=self.inf_factor)

    def add_paths_to_pool(self, data):
        utils.add_paths_to_pool(self.pool, data)

    def optimize_policy(self, itr, data):
        self.policy_opt.update_policy(data)


