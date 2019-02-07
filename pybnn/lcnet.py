from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from pybnn.bohamiann import Bohamiann

"""
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

"""


def vapor_pressure(x, a, b, c, *args):
    a_ = -a
    b_ = -b / 10.
    c_ = -c / 10.

    return (torch.exp(a_ + b_ / (x + 1e-5) + c_ * torch.log(x + 1e-5)) - torch.exp(a_ + b_))


def log_log_linear(x, a, b, *args):
    return torch.log(1 + a * torch.log(1 + x) + b)


def hill_3(x, a, b, c, *args):
    return a * (1. / ((c / x) ** b + 1.) - 1. / (c ** b + 1.))


def pow_func(x, a, b, *args):
    return a * ((x) ** b - 1)


def log_power(x, a, b, c, *args):
    b_n = b * 0.1
    return 2 * a * (1 / (1. + torch.exp(-c * b_n)) - 1 / (1. + (x / torch.exp(b_n)) ** c))


def pow_4(x, a, b, c, *args):
    a_n = a
    b_n = b / 10.
    c_n = c / 10.
    d = 1 + (a_n + b_n) ** (-c_n)
    return d - (a_n * x + b_n) ** (-c_n)


def mmf(x, a, b, c, d, *args):
    return a - (a - b) / (1 + (c * x) ** d)


def exp_4(x, a, b, c, d, *args):
    f_max = 1 - np.exp(-1)
    f_min = 0 - np.exp(1)
    return ((d - torch.exp(-a * x ** c + b)) - f_min) / (f_max - f_min)


def janoschek(x, a, b, c, d, *args):
    # c_n = c ** 2
    c_n = c
    return a - (a - b) * torch.exp(-c_n * x ** d)


def weibull(x, a, b, c, d, *args):
    # c_n = c ** 2
    return a - (a - b) * torch.exp(-(c * x) ** d)


def ilog_2(x, a, *args):
    a_n = a * 0.1
    b = a_n / np.log(2.1) + 1
    return b - a_n / torch.log((x + 1.1))


def linear(x, bias=0, a=1, *args):
    return a * x + bias


def quadratic(x, bias=0, a=0.1, b=0.1, *args):
    return a * (x ** 2) + b * x + bias


def exponential(x, a, b, *args):
    return b * (- torch.exp(-10 * a * x) + torch.exp(-10 * a))


def bf_layer(theta, t):
    y_a = vapor_pressure(t, theta[:, 0], theta[:, 1], theta[:, 2])
    y_b = log_log_linear(t, theta[:, 3], theta[:, 4])

    y_c = hill_3(t, theta[:, 5], theta[:, 6], theta[:, 7])

    y_d = pow_func(t, theta[:, 8], theta[:, 9])

    y_e = log_power(t, theta[:, 10], theta[:, 11], theta[:, 12])

    y_f = pow_4(t, theta[:, 13], theta[:, 14], theta[:, 15])

    y_g = mmf(t, theta[:, 16], theta[:, 17], theta[:, 18], theta[:, 19])

    y_h = exp_4(t, theta[:, 20], theta[:, 21], theta[:, 22], theta[:, 23])

    y_i = janoschek(t, theta[:, 24], theta[:, 25], theta[:, 26], theta[:, 27])

    y_j = weibull(t, theta[:, 28], theta[:, 29], theta[:, 30], theta[:, 31])

    y_k = ilog_2(t, theta[:, 32])

    y_l = linear(t, theta[:, 33], theta[:, 34])

    y_m = quadratic(t, theta[:, 35], theta[:, 36], theta[:, 37])

    y_n = exponential(t, theta[:, 38], theta[:, 39])

    return torch.stack([y_a, y_b, y_c, y_d, y_e, y_f, y_g, y_h, y_i, y_j, y_k, y_l, y_m, y_n], dim=1)


class AppendLayer(nn.Module):
    def __init__(self, bias=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if bias:
            self.bias = nn.Parameter(torch.DoubleTensor(1, 1))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return torch.cat((x, self.bias * torch.ones_like(x)), dim=1)


def get_lc_net_architecture(input_dimensionality: int) -> torch.nn.Module:
    class Architecture(nn.Module):
        def __init__(self, n_inputs, n_hidden=50):
            super(Architecture, self).__init__()
            self.fc1 = nn.Linear(n_inputs - 1, n_hidden)
            self.fc2 = nn.Linear(n_hidden, n_hidden)
            self.fc3 = nn.Linear(n_hidden, n_hidden)
            self.theta_layer = nn.Linear(n_hidden, 40)
            self.weight_layer = nn.Linear(n_hidden, 14)
            self.asymptotic_layer = nn.Linear(n_hidden, 1)
            # self.sigma_layer = nn.Linear(n_hidden, 1)
            self.sigma_layer = AppendLayer()

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

            return self.sigma_layer(mean)
            # std = torch.sigmoid(self.sigma_layer(x))

            # return torch.cat((mean, std), dim=1)

    return Architecture(n_inputs=input_dimensionality)


class LCNet(Bohamiann):
    def __init__(self,
                 metrics=(nn.MSELoss,)
                 ) -> None:
        super(LCNet, self).__init__(get_network=get_lc_net_architecture,
                                    normalize_input=True,
                                    normalize_output=False,
                                    metrics=metrics)

    def normalize_input(self, x, m=None, s=None):
        if m is None:
            m = np.mean(x, axis=0)
        if s is None:
            s = np.std(x, axis=0)

        x_norm = deepcopy(x)
        x_norm[:, :-1] = (x[:, :-1] - m[:-1]) / s[:-1]

        return x_norm, m, s
