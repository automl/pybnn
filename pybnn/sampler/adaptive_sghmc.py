import torch

from torch.optim import Optimizer


class AdaptiveSGHMC(Optimizer):
    """ Stochastic Gradient Hamiltonian Monte-Carlo Sampler that uses a burn-in
        procedure to adapt its own hyperparameters during the initial stages
        of sampling.

        See [1] for more details on this burn-in procedure.\n
        See [2] for more details on Stochastic Gradient Hamiltonian Monte-Carlo.

        [1] J. T. Springenberg, A. Klein, S. Falkner, F. Hutter
            In Advances in Neural Information Processing Systems 29 (2016).\n
            `Bayesian Optimization with Robust Bayesian Neural Networks. <http://aad.informatik.uni-freiburg.de/papers/16-NIPS-BOHamiANN.pdf>`_
        [2] T. Chen, E. B. Fox, C. Guestrin
            In Proceedings of Machine Learning Research 32 (2014).\n
            `Stochastic Gradient Hamiltonian Monte Carlo <https://arxiv.org/pdf/1402.4102.pdf>`_
    """

    def __init__(self,
                 params,
                 lr: float = 1e-2,
                 num_burn_in_steps: int = 3000,
                 epsilon: float = 1e-16,
                 mdecay: float = 0.05,
                 scale_grad: float = 1.) -> None:
        """ Set up a SGHMC Optimizer.

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr: float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        num_burn_in_steps: int, optional
            Number of burn-in steps to perform. In each burn-in step, this
            sampler will adapt its own internal parameters to decrease its error.
            Set to `0` to turn scale adaption off.
            Default: `3000`.
        epsilon: float, optional
            (Constant) per-parameter epsilon level.
            Default: `0.`.
        mdecay:float, optional
            (Constant) momentum decay per time-step.
            Default: `0.05`.
        scale_grad: float, optional
            Value that is used to scale the magnitude of the epsilon used
            during sampling. In a typical batches-of-data setting this usually
            corresponds to the number of examples in the entire dataset.
            Default: `1.0`.

        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(num_burn_in_steps))

        defaults = dict(
            lr=lr, scale_grad=float(scale_grad),
            num_burn_in_steps=num_burn_in_steps,
            mdecay=mdecay,
            epsilon=epsilon
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

                if len(state) == 0:
                    state["iteration"] = 0
                    state["tau"] = torch.ones_like(parameter)
                    state["g"] = torch.ones_like(parameter)
                    state["v_hat"] = torch.ones_like(parameter)
                    state["momentum"] = torch.zeros_like(parameter)
                state["iteration"] += 1

                mdecay, epsilon, lr = group["mdecay"], group["epsilon"], group["lr"]
                scale_grad = torch.tensor(group["scale_grad"], dtype=parameter.dtype)
                tau, g, v_hat = state["tau"], state["g"], state["v_hat"]

                momentum = state["momentum"]
                gradient = parameter.grad.data * scale_grad

                tau_inv = 1. / (tau + 1.)

                # update parameters during burn-in
                if state["iteration"] <= group["num_burn_in_steps"]:
                    tau.add_(- tau * (
                            g * g / (v_hat + epsilon)) + 1)  # specifies the moving average window, see Eq 9 in [1] left
                    g.add_(-g * tau_inv + tau_inv * gradient)  # average gradient see Eq 9 in [1] right
                    v_hat.add_(-v_hat * tau_inv + tau_inv * (gradient ** 2))  # gradient variance see Eq 8 in [1]

                minv_t = 1. / (torch.sqrt(v_hat) + epsilon)  # preconditioner

                epsilon_var = (2. * (lr ** 2) * mdecay * minv_t - (lr ** 4))

                # sample random epsilon
                sigma = torch.sqrt(torch.clamp(epsilon_var, min=1e-16))
                sample_t = torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient) * sigma)

                # update momentum (Eq 10 right in [1])
                momentum.add_(
                    - (lr ** 2) * minv_t * gradient - mdecay * momentum + sample_t
                )

                # update theta (Eq 10 left in [1])
                parameter.data.add_(momentum)

        return loss
