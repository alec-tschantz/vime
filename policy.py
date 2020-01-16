# pylint: disable=no-member
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable


class Policy(nn.Module):
    def __init__(self, obs_dim, hidden_dim, act_dim):
        super(Policy, self).__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, act_dim)

    def forward(self, x):
        y = torch.relu(self.fc1(x))
        y = torch.softmax(self.fc2(y), dim=-1)
        return y

    def get_action(self, state):
        state = torch.from_numpy(state).type(torch.FloatTensor)
        state = Variable(state)
        action_probs = self(state)
        distribution = Categorical(action_probs)
        action = distribution.sample()
        log_probs = distribution.log_prob(action)
        return int(action), log_probs


class PolicyOptimizer(object):

    def __init__(self, policy, gamma=0.99, lr=0.01):
        self.gamma = gamma
        self.policy = policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def update_policy(self, data):
        paths = data['paths']

        for path in paths:
            rewards = path['rewards']
            log_probs = path['log_probs']

            _rewards = []
            R = 0
            for r in rewards[::-1]:
                R = r + self.gamma * R
                _rewards.insert(0, R)

            _rewards = torch.FloatTensor(_rewards)
            _rewards = (_rewards - _rewards.mean()) / \
                (_rewards.std() + np.finfo(np.float32).eps)
            _rewards = Variable(_rewards)

            loss = (torch.sum(torch.mul(log_probs, _rewards).mul(-1), -1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
