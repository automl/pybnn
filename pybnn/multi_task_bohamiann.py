import logging
import time
import typing
from itertools import islice
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils

from pybnn.base_model import BaseModel
from pybnn.priors import weight_prior
from pybnn.sampler import AdaptiveSGHMC, SGLD, SGHMC, PreconditionedSGLD
from pybnn.util.infinite_dataloader import infinite_dataloader
from pybnn.util.normalization import zero_mean_unit_var_unnormalization, zero_mean_unit_var_normalization
from pybnn.bohamiann import Bohamiann


def get_default_network(input_dimensionality: int) -> torch.nn.Module:
    class AppendLayer(nn.Module):
        def __init__(self, bias=True, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if bias:
                self.bias = nn.Parameter(torch.DoubleTensor(1, 1))
            else:
                self.register_parameter('bias', None)

        def forward(self, x):
            return torch.cat((x, self.bias * torch.ones_like(x)), dim=1)

    def init_weights(module):
        if type(module) == AppendLayer:
            nn.init.constant_(module.bias, val=np.log(1e-3))
        elif type(module) == nn.Linear:
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="linear")
            nn.init.constant_(module.bias, val=0.0)

    return nn.Sequential(
        nn.Linear(input_dimensionality, 50), nn.Tanh(),
        nn.Linear(50, 50), nn.Tanh(),
        nn.Linear(50, 50), nn.Tanh(),
        nn.Linear(50, 1),
        AppendLayer()
    ).apply(init_weights)


def nll(input, target):
    batch_size = input.size(0)

    prediction_mean = input[:, 0].view((-1, 1))
    log_prediction_variance = input[:, 1].view((-1, 1))
    prediction_variance_inverse = 1. / (torch.exp(log_prediction_variance) + 1e-16)

    mean_squared_error = (target.view(-1, 1) - prediction_mean) ** 2

    log_likelihood = torch.sum(
        torch.sum(-mean_squared_error * (0.5 * prediction_variance_inverse) - 0.5 * log_prediction_variance, dim=1))

    log_likelihood = log_likelihood / batch_size

    return -log_likelihood


class MultiTaskBohamiann(Bohamiann):
    def __init__(self,
                 n_tasks: int,
                 get_network=get_default_network,
                 batch_size=20,
                 normalize_input: bool = True,
                 normalize_output: bool = True,
                 use_double_precision: bool = True,
                 sampling_method: str = "adaptive_sghmc",
                 metrics=(nn.MSELoss,)
                 ) -> None:
        """ Bayesian Neural Network for regression problems.

        Bayesian Neural Networks use Bayesian methods to estimate the posterior
        distribution of a neural network's weights. This allows to also
        predict uncertainties for test points and thus makes Bayesian Neural
        Networks suitable for Bayesian optimization.
        This module uses stochastic gradient MCMC methods to sample
        from the posterior distribution.

        See [1] for more details.

        [1] J. T. Springenberg, A. Klein, S. Falkner, F. Hutter
            Bayesian Optimization with Robust Bayesian Neural Networks.
            In Advances in Neural Information Processing Systems 29 (2016).

        Parameters
        ----------
        normalize_input: bool, optional
            Specifies if inputs should be normalized to zero mean and unit variance.
        normalize_output: bool, optional
            Specifies whether outputs should be un-normalized.
        """
        self.n_tasks = n_tasks
        super(MultiTaskBohamiann, self).__init__(get_network, batch_size, normalize_output, normalize_input,
                                                 sampling_method, metrics, use_double_precision)

    def normalize_input(self, x, m=None, s=None):
        if m is None:
            m = np.mean(x, axis=0)
        if s is None:
            s = np.std(x, axis=0)
        x_norm = deepcopy(x)
        x_norm[:, :-1] = (x[:, :-1] - m[:-1]) / s[:-1]

        return x_norm, m, s
