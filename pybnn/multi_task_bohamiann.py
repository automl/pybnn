from copy import deepcopy
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from pybnn.bohamiann import Bohamiann
from pybnn.util.layers import AppendLayer


def get_multitask_network(input_dimensionality: int, n_tasks: int) -> torch.nn.Module:

    class Architecture(torch.nn.Module):
        def __init__(self, n_inputs, n_tasks, emb_dim=5, n_hidden=50):
            super(Architecture, self).__init__()
            self.fc1 = torch.nn.Linear(n_inputs - 1 + emb_dim, n_hidden)
            self.fc2 = torch.nn.Linear(n_hidden, n_hidden)
            self.fc3 = torch.nn.Linear(n_hidden, 1)
            self.log_std = AppendLayer(noise=1e-3)
            self.emb = torch.nn.Embedding(n_tasks, emb_dim)
            self.n_tasks = n_tasks

        def forward(self, input):
            x = input[:, :-1]
            t = input[:, -1:]
            t_emb = self.emb(t.long()[:, 0])
            x = torch.cat((x, t_emb), dim=1)
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = self.fc3(x)
            return self.log_std(x)

    return Architecture(n_inputs=input_dimensionality, n_tasks=n_tasks)


class MultiTaskBohamiann(Bohamiann):
    def __init__(self,
                 n_tasks: int,
                 get_network=get_multitask_network,
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

        func = partial(get_network, n_tasks=n_tasks)
        super(MultiTaskBohamiann, self).__init__(func, normalize_output, normalize_input,
                                                 sampling_method, metrics, use_double_precision)

    def normalize_input(self, x, m=None, s=None):
        if m is None:
            m = np.mean(x, axis=0)
        if s is None:
            s = np.std(x, axis=0)
        x_norm = deepcopy(x)
        x_norm[:, :-1] = (x[:, :-1] - m[:-1]) / s[:-1]

        return x_norm, m, s
