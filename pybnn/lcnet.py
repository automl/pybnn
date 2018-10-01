import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy

from pybnn.bohamiann import Bohamiann


def vapor_pressure(t, a, b, c):
    a_ = a
    b_ = b / 10.
    c_ = c / 10.

    return torch.exp(-a_ - b_ / t - c_ * torch.log(t)) - torch.exp(-a_ - b_)


def pow_func(t, a, b):
    return a * (t ** b - 1)


def log_power(t, a, b, c):
    b_ = b / 10

    return 2 * a * (1 / (1 + torch.exp(-b_ * c)) - 1 / (1 + torch.exp(c * torch.log(t) - b_)))


def exponential(t, a, b):
    b_ = 10 * b
    return a * (torch.exp(-b_) - torch.exp(-b_ * t))


def hill_3(t, a, b, c):
    return a * (1 / (c ** b * t ** (-b) + 1) - a / (c ** b + 1))


def bf_layer(theta, t):
    a = theta[:, 0]
    b = theta[:, 1]
    c = theta[:, 2]
    y_a = vapor_pressure(t, a, b, c)

    a = theta[:, 3]
    b = theta[:, 4]
    y_b = pow_func(t, a, b)

    a = theta[:, 5]
    b = theta[:, 6]
    c = theta[:, 7]
    y_c = log_power(t, a, b, c)

    a = theta[:, 8]
    b = theta[:, 9]
    y_d = exponential(t, a, b)

    a = theta[:, 10]
    b = theta[:, 11]
    c = theta[:, 12]
    y_e = hill_3(t, a, b, c)

    return torch.stack([y_a, y_b, y_c, y_d, y_e], dim=1)


def get_lc_net_architecture(input_dimensionality: int) -> torch.nn.Module:
    class Architecture(nn.Module):
        def __init__(self, n_inputs, n_hidden=50):
            super(Architecture, self).__init__()
            self.fc1 = nn.Linear(n_inputs - 1, n_hidden)
            self.fc2 = nn.Linear(n_hidden, n_hidden)
            self.fc3 = nn.Linear(n_hidden, n_hidden)
            self.theta_layer = nn.Linear(n_hidden, 13)
            self.weight_layer = nn.Linear(n_hidden, 5)
            self.asymptotic_layer = nn.Linear(n_hidden, 1)
            self.sigma_layer = nn.Linear(n_hidden, 1)

        def forward(self, input):
            x = input[:, :-1]
            t = input[:, -1]

            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            theta = torch.sigmoid(self.theta_layer(x))

            bf = bf_layer(theta, t)
            weights = torch.softmax(self.weight_layer(x), -1)

            residual = torch.sum(bf * weights, dim=(1,), keepdim=True)

            asymptotic = torch.sigmoid(self.asymptotic_layer(x))

            mean = residual + asymptotic

            std = torch.sigmoid(self.sigma_layer(x))

            return torch.cat((mean, std), dim=1)

    return Architecture(n_inputs=input_dimensionality)


class LCNet(Bohamiann):
    def __init__(self,
                 batch_size=20,
                 metrics=(nn.MSELoss,)
                 ) -> None:
        super(LCNet, self).__init__(get_network=get_lc_net_architecture,
                                    batch_size=batch_size,
                                    normalize_input=True,
                                    normalize_output=False,
                                    metrics=metrics)

    def normalize(self, x, m=None, s=None):
        if m is None:
            m = np.mean(x, axis=0)
        if s is None:
            s = np.std(x, axis=0)

        x_norm = deepcopy(x)
        x_norm[:, :-1] = (x[:, :-1] - m[:-1]) / s[:-1]

        return x_norm, m, s
