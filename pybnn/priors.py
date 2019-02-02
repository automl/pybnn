from typing import Iterable
import numpy as np
import torch


def log_variance_prior(log_variance: torch.Tensor, mean: float = 1e-6, variance: float = 0.01) -> torch.Tensor:
    return torch.mean(
        torch.sum(
            ((-((log_variance - torch.log(torch.tensor(mean, dtype=log_variance.dtype))) ** 2)) /
             (2. * variance)) - 0.5 * torch.log(torch.tensor(variance, dtype=log_variance.dtype)),
            dim=1
        )
    )


def weight_prior(parameters: Iterable[torch.Tensor], dtype=np.float64, wdecay: float = 1.) -> torch.Tensor:

    num_parameters = 0
    log_likelihood = torch.from_numpy(np.array(0, dtype=dtype))
    for parameter in parameters:
        num_parameters += parameter.numel()
        log_likelihood += torch.sum(-wdecay * 0.5 * (parameter ** 2))

    return log_likelihood / num_parameters
