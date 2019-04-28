from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from pybnn.bohamiann import Bohamiann
from pybnn.util.layers import AppendLayer


def vapor_pressure(x, a, b, c, *args):
    b_ = (b + 1) / 2 / 10
    a_ = (a + 1) / 2
    c_ = (c + 1) / 2 / 10
    return torch.exp(-a_ - b_ / (x + 1e-5) - c_ * torch.log(x)) - (torch.exp(a_ + b_))


def log_func(t, a, b, c, *args):
    a_ = (a + 1) / 2 * 5
    b_ = (b + 1) / 2
    c_ = (c + 1) / 2 * 10
    return (c_ + a_ * torch.log(b_ * t + 1e-10)) / 10.


def hill_3(x, a, b, c, *args):
    a_ = (a + 1) / 2
    b_ = (b + 1) / 2
    c_ = (c + 1) / 2 / 100
    return a_ * (1. / ((c_ / x + 1e-5) ** b_ + 1.))


def bf_layer(theta, t):
    y_a = vapor_pressure(t, theta[:, 0], theta[:, 1], theta[:, 2])

    y_b = log_func(t, theta[:, 3], theta[:, 4], theta[:, 5])

    y_c = hill_3(t, theta[:, 6], theta[:, 7], theta[:, 8])

    return torch.stack([y_a, y_b, y_c], dim=1)


def get_lc_net_architecture(input_dimensionality: int) -> torch.nn.Module:
    class Architecture(nn.Module):
        def __init__(self, n_inputs, n_hidden=50):
            super(Architecture, self).__init__()
            self.fc1 = nn.Linear(n_inputs - 1, n_hidden)
            self.fc2 = nn.Linear(n_hidden, n_hidden)
            self.fc3 = nn.Linear(n_hidden, n_hidden)
            self.theta_layer = nn.Linear(n_hidden, 9)
            self.weight_layer = nn.Linear(n_hidden, 3)
            self.asymptotic_layer = nn.Linear(n_hidden, 1)
            self.sigma_layer = AppendLayer(noise=1e-3)

        def forward(self, input):
            x = input[:, :-1]
            t = input[:, -1]
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            theta = torch.tanh(self.theta_layer(x))

            bf = bf_layer(theta, t)
            weights = torch.softmax(self.weight_layer(x), -1)
            residual = torch.tanh(torch.sum(bf * weights, dim=(1,), keepdim=True))

            asymptotic = torch.sigmoid(self.asymptotic_layer(x))

            mean = residual + asymptotic
            return self.sigma_layer(mean)

    return Architecture(n_inputs=input_dimensionality)


class LCNet(Bohamiann):
    def __init__(self, **kwargs) -> None:
        super(LCNet, self).__init__(get_network=get_lc_net_architecture,
                                    normalize_input=True,
                                    normalize_output=False,
                                    **kwargs)

    @staticmethod
    def normalize_input(x, m=None, s=None):
        if m is None:
            m = np.mean(x, axis=0)
        if s is None:
            s = np.std(x, axis=0)

        x_norm = deepcopy(x)
        x_norm[:, :-1] = (x[:, :-1] - m[:-1]) / s[:-1]

        return x_norm, m, s
