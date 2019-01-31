from typing import Iterable

import torch


def log_variance_prior(log_variance: torch.Tensor, mean: float = 1e-6, variance: float = 0.01) -> torch.Tensor:
    return torch.mean(
        torch.sum(
            ((-((log_variance - torch.log(torch.tensor(mean, dtype=log_variance.dtype))) ** 2)) /
             (2. * variance)) - 0.5 * torch.log(torch.tensor(variance, dtype=log_variance.dtype)),
            dim=1
        )
    )


def weight_prior(parameters: Iterable[torch.Tensor], wdecay: float = 1.) -> torch.Tensor:
    num_parameters = torch.sum(torch.tensor([
        torch.prod(torch.tensor(parameter.size()))
        for parameter in parameters
    ]))

    log_likelihood = torch.sum(torch.tensor([
        torch.sum(-wdecay * 0.5 * (parameter ** 2))
        for parameter in parameters
    ]))

    return log_likelihood / (num_parameters.float() + 1e-16)
