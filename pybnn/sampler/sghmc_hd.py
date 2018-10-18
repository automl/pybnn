# vim: foldmethod=marker
from collections import OrderedDict
import typing
import numpy as np
import sympy
import torch
from torch.optim import Optimizer


def heaviside(x):
    return (x < 0).double() * torch.zeros_like(x) + (x >= 0).double() * torch.ones_like(x)

def maximum(a, b):
    return (a >= b).double() * a + (a < b).double() * b


def sympy_derivative(with_respect_to: str, tensor_names: typing.List[str], delta=1e-7):
    assert with_respect_to in tensor_names
    symbols = OrderedDict((
        (name, sympy.symbols(name))
        for name in tensor_names
    ))

    g2, p = symbols["g2"], symbols["p"]
    epsilon, mdecay, noise = symbols["epsilon"], symbols["mdecay"], symbols["noise"]
    grad, scale_grad = symbols["grad"], symbols["scale_grad"]
    random_sample = symbols["random_sample"]

    Minv = 1. / (sympy.sqrt(g2 + delta) + delta)

    epsilon_scaled = epsilon / sympy.sqrt(scale_grad)
    noise_scale = 2. * (epsilon_scaled ** 2) * mdecay * Minv - 2. * epsilon_scaled ** 3 * (Minv ** 2) * noise
    sigma = sympy.sqrt(sympy.Max(noise_scale, delta))

    sample_t = random_sample * sigma
    p_t = p - (epsilon ** 2) * Minv * grad - mdecay * p + sample_t

    derivative = sympy.diff(
        p_t, symbols[with_respect_to]
    )

    # Callable derivative function that computes d_theta d_`with respect to`
    return sympy.lambdify(
        args=tuple(symbols.values()),
        expr=derivative,
        modules={"sqrt": torch.sqrt, "Heaviside": heaviside, "Max": maximum}
    )


class SGHMCHD(Optimizer):
    name = "SGHMCHD"

    def __init__(self,
                 params,
                 hyper_lr: np.float64=1e-3,
                 lr: np.float64=1e-2,
                 num_burn_in_steps: int=3000,
                 noise: np.float64=0.,
                 mdecay: np.float64=0.05,
                 scale_grad: np.float64=1.) -> None:
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(num_burn_in_steps))

        defaults = dict(
            lr=lr, hyper_lr=hyper_lr, scale_grad=float(scale_grad),
            num_burn_in_steps=num_burn_in_steps,
            mdecay=mdecay,
            noise=noise
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

                #  State initialization {{{ #

                if len(state) == 0:
                    state["iteration"] = 0
                    state["tau"] = torch.ones_like(parameter)
                    state["g"] = torch.ones_like(parameter)
                    state["v_hat"] = torch.ones_like(parameter)
                    state["momentum"] = torch.zeros_like(parameter)
                    state["lr"] = torch.tensor(
                        torch.ones_like(parameter) * group["lr"], requires_grad=True
                    )

                    state["dxdh"] = torch.zeros_like(parameter)

                    state["dxdh_function"] = sympy_derivative(
                        with_respect_to="epsilon",
                        tensor_names=(
                            "g2", "p", "epsilon", "mdecay", "noise",
                            "grad", "scale_grad", "random_sample"
                        )
                    )

                    import numpy as np
                    from torch.optim import Adamax
                    state["hyperoptimizer"] = Adamax(
                        params=(state["lr"],),
                        lr=0.002 / np.sqrt(group["scale_grad"])
                    )
                state["iteration"] += 1
                #  }}} State initialization #

                #  Readability {{{ #
                mdecay, noise, lr = group["mdecay"], group["noise"], state["lr"]
                scale_grad = torch.tensor(group["scale_grad"]).double()

                tau, g, v_hat = state["tau"], state["g"], state["v_hat"]
                momentum = state["momentum"]

                gradient = parameter.grad.data
                #  }}} Readability #

                random_sample = torch.normal(mean=0., std=torch.ones_like(parameter)).double()

                #  Hypergradient Update {{{ #
                # Update derivative of parameters with respect to hyperparameter
                torch_tensors = (
                    v_hat, momentum, lr, mdecay, noise,
                    gradient, scale_grad, random_sample

                )

                dxdh_t = state["dxdh_function"](*torch_tensors)

                state["hyperoptimizer"].zero_grad()
                dxdlr = gradient * state["dxdh"]
                # XXX: How to use other hyperloss functions except nll here?
                state["lr"].grad = dxdlr
                state["lr"].grad.data = dxdlr
                state["hyperoptimizer"].step()

                #  }}} Hypergradient Update #

                r_t = 1. / (tau + 1.)
                minv_t = 1. / torch.sqrt(v_hat)

                #  Burn-in updates {{{ #
                if state["iteration"] <= group["num_burn_in_steps"]:
                    # Update state
                    tau.add_(1. - tau * (g * g / v_hat))
                    g.add_(-g * r_t + r_t * gradient)
                    v_hat.add_(-v_hat * r_t + r_t * (gradient ** 2))
                #  }}} Burn-in updates #

                lr_scaled = lr / torch.sqrt(scale_grad)

                #  Draw random sample {{{ #

                noise_scale = (
                    2. * (lr_scaled ** 2) * mdecay * minv_t -
                    2. * (lr_scaled ** 3) * (minv_t ** 2) * noise -
                    (lr_scaled ** 4)
                )

                sigma = torch.sqrt(torch.clamp(noise_scale, min=1e-16))

                # sample_t = torch.normal(mean=0., std=sigma)
                sample_t = random_sample * sigma

                #  }}} Draw random sample #

                #  SGHMC Update {{{ #
                momentum_t = momentum.add_(
                    - (lr ** 2) * minv_t * gradient - mdecay * momentum + sample_t
                )

                parameter.data.add_(momentum_t)

                #  }}} SGHMC Update #

                # Update gradient of parameters with respect to hyperparameter.
                state["dxdh"].copy_(dxdh_t)

        return loss
