import gym
import numpy as np
from bnn import BNN
from feedforward import FeedforwardModel
from policy import Policy, PolicyOptimizer
from experiment import Experiment


def run_cartpole():
    env = gym.make('CartPole-v0')

    obs_dim = np.prod(env.observation_space.shape)
    n_actions = env.action_space.n

    policy_hidden_dim = 256
    policy = Policy(obs_dim, policy_hidden_dim, n_actions)

    exp = Experiment(policy, None, env, exp_name="cartpole_basic", train_model=False, calc_inf_gain=False)
    exp.train()


def run_cartpole_expl():
    env = gym.make('CartPole-v0')

    obs_dim = np.prod(env.observation_space.shape)
    act_dim = np.prod(env.action_space.shape)
    n_actions = env.action_space.n

    policy_hidden_dim = 256
    policy = Policy(obs_dim, policy_hidden_dim, n_actions)

    input_dim = int(obs_dim + act_dim)
    output_dim = int(obs_dim)
    hidden_dim = 64
    model = BNN(input_dim, hidden_dim, output_dim)

    exp = Experiment(policy, model, env, exp_name="cartpole_expl", train_model=True, calc_inf_gain=True)
    exp.train()


if __name__ == "__main__":
    run_cartpole_expl()
