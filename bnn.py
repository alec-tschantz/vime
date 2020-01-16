# pylint: disable=maybe-no-member

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class BNNLayer(nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 std_prior):
        super(BNNLayer, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.std_prior = std_prior
        self.rho_prior = self.std_to_log(self.std_prior)

        self.W = np.random.normal(
            0., std_prior, (self.n_inputs, self.n_outputs))
        self.b = np.zeros((self.n_outputs,), dtype=np.float)

        W_mu = torch.Tensor(self.n_inputs, self.n_outputs)
        W_mu = nn.init.normal_(W_mu, mean=0., std=1.)
        self.W_mu = nn.Parameter(W_mu)

        W_rho = torch.Tensor(self.n_inputs, self.n_outputs)
        W_rho = nn.init.constant_(W_rho, self.rho_prior)
        self.W_rho = nn.Parameter(W_rho)

        b_mu = torch.Tensor(self.n_outputs, )
        b_mu = nn.init.normal_(b_mu, mean=0., std=1.)
        self.b_mu = nn.Parameter(b_mu)

        b_rho = torch.Tensor(self.n_outputs,)
        b_rho = nn.init.constant_(b_rho, self.rho_prior)
        self.b_rho = nn.Parameter(b_rho)

        self.W_mu_old = torch.Tensor(self.n_inputs, self.n_outputs).detach()
        self.W_rho_old = torch.Tensor(self.n_inputs, self.n_outputs).detach()
        self.b_mu_old = torch.Tensor(self.n_outputs,).detach()
        self.b_rho_old = torch.Tensor(self.n_outputs,).detach()

    def forward(self, X, infer=False):
        if infer:
            output = torch.mm(X, self.W_mu) + \
                self.b_mu.expand(X.size()[0], self.n_outputs)
            return output

        W = self.get_W()
        b = self.get_b()
        output = torch.mm(X, W) + b.expand(X.size()
                                           [0], self.n_outputs)
        return output

    def get_W(self):
        epsilon = torch.Tensor(self.n_inputs, self.n_outputs)
        epsilon = nn.init.normal_(epsilon, mean=0., std=1.)
        epsilon = Variable(epsilon)
        self.W = self.W_mu + self.log_to_std(self.W_rho) * epsilon
        return self.W

    def get_b(self):
        epsilon = torch.Tensor(self.n_outputs, )
        epsilon = nn.init.normal_(epsilon, mean=0., std=1.)
        epsilon = Variable(epsilon)
        self.b = self.b_mu + self.log_to_std(self.b_rho) * epsilon
        return self.b

    def log_to_std(self, rho):
        return torch.log(1 + torch.exp(rho))

    def std_to_log(self, std):
        return np.log(np.exp(std) - 1)

    def kl_div_new_prior(self):
        kl_div = self.kl_div(
            self.W_mu, self.log_to_std(self.W_rho), 0., self.std_prior)
        kl_div += self.kl_div(self.b_mu,
                              self.log_to_std(self.b_rho), 0., self.std_prior)
        return kl_div

    def kl_div_new_old(self):
        kl_div = self.kl_div(
            self.W_mu, self.log_to_std(self.W_rho), self.W_mu_old, self.log_to_std(self.W_rho_old))
        kl_div += self.kl_div(
            self.b_mu, self.log_to_std(self.b_rho), self.b_mu_old, self.log_to_std(self.b_rho_old))
        return kl_div

    def kl_div(self, p_mean, p_std, q_mean, q_std):
        #TODO:
        if not hasattr(q_std, 'data'):
            torch_q_std = torch.Tensor([q_std])
        else:
            torch_q_std = q_std
        numerator = (p_mean - q_mean)**2 + p_std**2 - q_std**2
        denominator = 2 * q_std**2 + 1e-8
        return((numerator / denominator + torch.log(torch_q_std) - torch.log(p_std)).sum())

    def save_old_params(self):
        self.W_mu_old = self.W_mu.clone()
        self.W_rho_old = self.W_rho.clone()
        self.b_mu_old = self.b_mu.clone()
        self.b_rho_old = self.b_rho.clone()

    def reset_to_old_params(self):
        self.W_mu.data = self.W_mu_old.data
        self.W_rho.data = self.W_rho_old.data
        self.b_mu.data = self.b_mu_old.data
        self.b_rho.data = self.b_rho_old.data


class BNN(nn.Module):
    def __init__(self, n_input,
                 n_hidden,
                 n_output,
                 n_batches=5,
                 std_prior=0.5,
                 std_likelihood=5.0,
                 n_samples=10,
                 learning_rate=0.0001):
        super(BNN, self).__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.std_prior = std_prior

        self.std_prior = std_prior
        self.std_likelihood = std_likelihood
        self.n_samples = n_samples
        self.learning_rate = learning_rate
        self.n_batches = n_batches

        self.l1 = BNNLayer(n_input, n_hidden, std_prior)
        self.l2 = BNNLayer(n_hidden, n_output, std_prior)
        self.layers = [self.l1, self.l2]

        self.opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, X, infer=False):
        output = torch.relu(
            self.l1(X, infer))
        output = self.l2(output, infer)
        return output

    def loss(self, input, target):
        _log_p_D_given_W = []
        for _ in range(self.n_samples):
            prediction = self(input)
            _log_p_D_given_W.append(self._log_prob_normal(
                target, prediction, self.std_likelihood))

        kl = self.kl_div_new_prior()
        return kl / self.n_batches - sum(_log_p_D_given_W) / self.n_samples

    def loss_last_sample(self, input, target):
        _log_p_D_given_w = []
        for _ in range(self.n_samples):
            prediction = self(input)
            _log_p_D_given_w.append(self._log_prob_normal(
                target, prediction, self.std_likelihood))

        return self.kl_div_new_old() - sum(_log_p_D_given_w) / self.n_samples

    def train_fn(self, inputs, targets):
        inputs = torch.from_numpy(inputs).float()
        targets = torch.from_numpy(targets).float()

        self.opt.zero_grad()
        loss = self.loss(inputs, targets)
        loss.backward()
        self.opt.step()

        return loss.item()

    def train_update_fn(self, inputs, targets):
        inputs = torch.from_numpy(inputs).float()
        targets = torch.from_numpy(targets).float()

        self.opt.zero_grad()
        loss = self.loss_last_sample(inputs, targets)
        loss.backward()
        self.opt.step()

        return loss.item()

    def pred_fn(self, inputs):
        inputs = torch.from_numpy(inputs).float()
        with torch.no_grad():
            _out = self(inputs, infer=True)
        return _out

    def kl_div_new_old(self):
        """KL divergence KL[new_parans||old_param]"""
        kl_divs = [l.kl_div_new_old() for l in self.layers]
        return sum(kl_divs)

    def kl_div_new_prior(self):
        """KL divergence KL[new_parans||prior]"""
        kl_divs = [l.kl_div_new_prior() for l in self.layers]
        return sum(kl_divs)

    def save_old_params(self):
        for l in self.layers:
            l.save_old_params()

    def reset_to_old_params(self):
        for l in self.layers:
            l.reset_to_old_params()

    def _log_prob_normal(self, input, mu=0., sigma=1.):
        sigma = torch.Tensor([sigma])
        log_sigma = torch.log(sigma)
        log_normal = - log_sigma - np.log(np.sqrt(2 * np.pi)) - \
            (input - mu)**2 / (2 * (sigma**2))
        return torch.sum(log_normal)
