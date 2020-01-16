# pylint: disable=E1101
import numpy as np
import torch
import torch.nn as nn


class FeedforwardModel(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super(FeedforwardModel, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.mse_loss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def train_fn(self, inputs, targets):
        inputs = self.to_tensor(inputs)
        targets = self.to_tensor(targets)

        self.opt.zero_grad()
        y_hat = self(inputs)
        loss = self.mse_loss(y_hat, targets)
        loss.backward()
        self.opt.step()

    def pred_fn(self, inputs):
        inputs = self.to_tensor(inputs)
        with torch.no_grad():
            _out = self(inputs)
        return _out

    def to_tensor(self, data):
        tensor = torch.from_numpy(data)  
        tensor = tensor.float()
        return tensor
