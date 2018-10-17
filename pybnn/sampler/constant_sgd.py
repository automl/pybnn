import torch
from torch.optim import Optimizer


class ConstantSGD(Optimizer):
    def __init__(self,
                 params,
                 batch_size,
                 num_data_points,
                 precondition_decay_rate=0.95) -> None:
        """

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.

        precondition_decay_rate : float, optional
            Exponential decay rate of the rescaling of the preconditioner (RMSprop).
            Should be smaller than but nearly `1` to approximate sampling from the posterior.
            Default: `0.95`

        """
        self.batch_size = batch_size
        self.num_data_points = num_data_points
        self.precondition_decay_rate = precondition_decay_rate
        defaults = dict(precondition_decay_rate=precondition_decay_rate)
        super().__init__(params, defaults)

    def step(self, closure=None, noise=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]

                if len(state) == 0:
                    state["iteration"] = 0
                    state["epsilon"] = torch.zeros(1)
                    state["v"] = torch.ones_like(parameter).double()
                gradient = parameter.grad.data
                state["iteration"] += 1
                precondition_decay_rate = group["precondition_decay_rate"]

                v = state["v"]
                v_t = v * precondition_decay_rate + (1.0 - precondition_decay_rate) * (gradient ** 2)
                state["v"] = v_t

                # if noise is None:
                epsilon = (2 * self.batch_size) / (self.num_data_points * torch.sqrt(v_t)) # TODO: variance of std???
                # else:
                #     epsilon = 2 * self.batch_size / (self.num_data_points * noise)
                state["epsilon"] = epsilon

                parameter.data.add_(-epsilon * gradient)

        return loss
