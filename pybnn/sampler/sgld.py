import numpy as np
import torch
from torch.optim import Optimizer


# def decay(t, a, b , gamma):
#     return a * (b + t) ** (-gamma)
#
# def poly(base_lr, t, max_iter, power):
#     return base_lr * (1 - t / max_iter) ** power
#
# def const(*args):
#     return 1


class SGLD(Optimizer):
    """ Stochastic Gradient Langevin Dynamics Sampler
    """

    def __init__(self,
                 params,
                 lr: np.float64 = 1e-2,
                 scale_grad: np.float64 = 1) -> None:

        """ Set up a SGLD Optimizer.

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr : float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        # if lr_decay is None:
        #     self.lr_decay = const
        #     pass
        # elif lr_decay == "inv":
        #     final_lr_fraction = 1e-2
        #     degree = 2
        #     gamma = (np.power(1 / final_lr_fraction, 1. / degree) - 1) / (T - 1)
        #     self.lr_decay = lambda t: lr * np.power((1 + gamma * t), -degree)
        # else:
        #     self.lr_decay = lr_decay
        defaults = dict(
            lr=lr,
            scale_grad=scale_grad
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                lr, scale_grad = group["lr"], group["scale_grad"]
                # the average gradient over the batch, i.e N/n sum_i g_theta_i + g_prior
                gradient = parameter.grad.data * scale_grad
                #  State initialization
                if len(state) == 0:
                    state["iteration"] = 0

                sigma = torch.sqrt(torch.from_numpy(np.array(lr, dtype=type(lr))))
                delta = (0.5 * lr * gradient +
                         sigma * torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)))

                parameter.data.add_(-delta)
                state["iteration"] += 1
                state["sigma"] = sigma

        return loss
