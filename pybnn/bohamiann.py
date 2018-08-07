from itertools import islice
import logging
import typing

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils

from pybnn.base_model import BaseModel
from pybnn.util.normalization import zero_mean_unit_var_unnormalization, zero_mean_unit_var_normalization
from pybnn.util.infinite_dataloader import infinite_dataloader
from pybnn.sampler.adaptive_sghmc import AdaptiveSGHMC


def get_network(input_dimensionality: int) -> torch.nn.Module:
    class AppendLayer(nn.Module):
        def __init__(self, bias=True, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if bias:
                self.bias = nn.Parameter(torch.Tensor(1, 1))
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
        AppendLayer()).apply(init_weights)


def nll(input, target):
    batch_size = input.size(0)

    prediction_mean = input[:, 0].view((-1, 1))
    log_prediction_variance = input[:, 1].view((-1, 1))
    prediction_variance_inverse = 1. / (torch.exp(log_prediction_variance) + 1e-16)

    mean_squared_error = (target.view(-1, 1) - prediction_mean) ** 2

    log_likelihood = torch.sum(
        torch.sum(-mean_squared_error * (0.5 * prediction_variance_inverse) - 0.5 * log_prediction_variance, dim=1))

    log_likelihood = log_likelihood / batch_size

    # TODO: add priors
    #    log_likelihood += (
    #            self.log_variance_prior(log_prediction_variance) / self.num_datapoints
    #    )
    #
    #    log_likelihood += self.weight_prior(self.parameters) / self.num_datapoints

    return -log_likelihood


class Bohamiann(BaseModel):
    def __init__(self,
                 batch_size=20,
                 normalize_input: bool = True,
                 normalize_output: bool = True,
                 num_steps: int = 13000,
                 burn_in_steps: int = 3000,
                 keep_every: int = 100,
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
            Specifies whether outputs should be unnormalized.
        num_steps: int, optional
            Number of sampling steps to perform after burn-in is finished.
            In total, `num_steps // keep_every` network weights will be sampled.
            Defaults to `10000`.
        burn_in_steps: int, optional
            Number of burn-in steps to perform.
            This value is passed to the given `optimizer` if it supports special
            burn-in specific behavior.
            Networks sampled during burn-in are discarded.
            Defaults to `3000`.
        keep_every: int, optional
            Number of sampling steps (after burn-in) to perform before keeping a sample.
            In total, `num_steps // keep_every` network weights will be sampled.
            Defaults to `100`.
        """

        assert burn_in_steps >= 0, "Invalid value for amount of burn-in steps -- cannot be negative."
        assert keep_every >= 1, "Invalid value for `keep_every`. Specify how many " \
                                "sampling steps to perform before keeping a sample."
        assert num_steps > burn_in_steps + keep_every, "Not even a single network would be sampled."
        assert batch_size >= 1, "Invalid batch size. Batches must contain at least a single sample."

        self.batch_size = batch_size

        self.num_steps = num_steps
        self.num_burn_in_steps = burn_in_steps
        self.metrics = metrics
        self.keep_every = keep_every
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output

        self.sampled_weights = []  # type: typing.List[typing.Tuple[np.ndarray]]

        logging.info("Performing %d iterations" % (self.num_steps))

    def _keep_sample(self, step: int) -> bool:
        """ Determine if the network weight sample recorded at `step` should be stored.
            Samples are recorded after burn-in (`step > self.num_burn_in_steps`),
            and only every `self.keep_every` th step.

        Parameters
        ----------
        step: int
            Current iteration count.

        Returns
        ----------
        should_keep: bool
            Sentinel that is `True` if and only if network weights should be stored at `step`.

        """
        if step < self.num_burn_in_steps:
            logging.debug("Skipping burn-in sample, step = %d" % step)
            return False
        sample_t = step - self.num_burn_in_steps
        return sample_t % self.keep_every == 0

    @property
    def network_weights(self) -> np.ndarray:
        """ Extract current network weight values as `np.ndarray`.

        Returns
        ----------
        weight_values: np.ndarray
            Numpy array containing current network weight values.

        """
        return tuple(
            np.asarray(torch.tensor(parameter.data).numpy())
            for parameter in self.model.parameters()
        )

    @network_weights.setter
    def network_weights(self, weights: typing.List[np.ndarray]) -> None:
        """ Assign new `weights` to our neural networks parameters.

        Parameters
        ----------
        weights : typing.List[np.ndarray]
            List of weight values to assign.
            Individual list elements must have shapes that match
            the network parameters with the same index in `self.network_weights`.

        Examples
        ----------
        This serves as a handy bridge between our pytorch parameters
        and corresponding values for them represented as numpy arrays:

        >>> import numpy as np
        >>> bnn = BayesianNeuralNetwork()
        >>> input_dimensionality = 1
        >>> bnn.model = bnn.network_architecture(input_dimensionality)
        >>> dummy_weights = [np.random.rand(parameter.shape) for parameter in bnn.model.parameters()]
        >>> bnn.network_weights = dummy_weights
        >>> np.allclose(bnn.network_weights, dummy_weights)
        True

        """
        logging.debug("Assigning new network weights: %s" % str(weights))
        for parameter, sample in zip(self.model.parameters(), weights):
            parameter.copy_(torch.from_numpy(sample))

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        """ Train a BNN using input datapoints `x_train` with corresponding labels `y_train`.
        Parameters
        ----------
        x_train : numpy.ndarray (N, D)
            Input training datapoints.
        y_train : numpy.ndarray (N,)
            Input training labels.
        """
        logging.debug("Training started.")

        logging.debug("Clearing list of sampled weights.")
        self.sampled_weights.clear()

        num_datapoints, input_dimensionality = x_train.shape
        logging.debug(
            "Processing %d training datapoints "
            " with % dimensions each." % (num_datapoints, input_dimensionality)
        )

        x_train_ = np.asarray(x_train)

        if self.normalize_input:
            logging.debug(
                "Normalizing training datapoints to "
                " zero mean and unit variance."
            )
            x_train_, self.x_mean, self.x_std = zero_mean_unit_var_normalization(x_train)

        y_train_ = np.asarray(y_train)

        if self.normalize_output:
            logging.debug("Normalizing training labels to zero mean and unit variance.")
            y_train_, self.y_mean, self.y_std = zero_mean_unit_var_normalization(y_train)

        train_loader = infinite_dataloader(
            data_utils.DataLoader(
                data_utils.TensorDataset(
                    torch.from_numpy(x_train_).float(),
                    torch.from_numpy(y_train_).float()
                ),
                batch_size=self.batch_size
            )
        )

        self.model = get_network(input_dimensionality=input_dimensionality)

        sampler = AdaptiveSGHMC(self.model.parameters(), scale_grad=num_datapoints)

        batch_generator = islice(enumerate(train_loader), self.num_steps)

        for epoch, (x_batch, y_batch) in batch_generator:
            sampler.zero_grad()
            loss = nll(input=self.model(x_batch), target=y_batch)
            loss.backward()
            sampler.step()

            if self._keep_sample(epoch):
                logging.debug("Recording sample, epoch = %d " % (epoch))
                weights = self.network_weights
                logging.debug("Sampled weights:\n%s" % str(weights))
                self.sampled_weights.append(weights)

        self.is_trained = True
        return self

    def predict(self, x_test: np.ndarray, return_individual_predictions: bool = False):
        logging.debug("Predicting started.")
        x_test_ = np.asarray(x_test)

        logging.debug(
            "Processing %d test datapoints "
            " with %d dimensions each." % x_test_.shape
        )

        if self.normalize_input:
            logging.debug(
                "Normalizing test datapoints to "
                " zero mean and unit variance."
            )
            x_test_, *_ = zero_mean_unit_var_normalization(x_test, self.x_mean, self.x_std)

        def network_predict(x_test_, weights):
            logging.debug(
                "Predicting on data:\n%s Using weights:\n%s" % (
                    str(x_test_), str(weights)
                )
            )
            with torch.no_grad():
                self.network_weights = weights
                return self.model(torch.from_numpy(x_test_).float()).numpy()[:, 0]

        logging.debug("Predicting with %d networks." % len(self.sampled_weights))
        network_outputs = [
            network_predict(x_test_, weights=weights)
            for weights in self.sampled_weights
        ]

        mean_prediction = np.mean(network_outputs, axis=0)
        variance_prediction = np.mean((network_outputs - mean_prediction) ** 2, axis=0)

        if self.normalize_output:
            logging.debug("Unnormalizing predictions.")
            logging.debug(
                "Mean of network predictions "
                "before unnormalization:\n%s" % str(mean_prediction)
            )
            logging.debug(
                "Variance/Uncertainty of network predictions "
                "before unnormalization:\n%s" % str(variance_prediction)
            )

            mean_prediction = zero_mean_unit_var_unnormalization(
                mean_prediction, self.y_mean, self.y_std
            )
            variance_prediction *= self.y_std ** 2

            logging.debug(
                "Mean of network predictions "
                "after unnormalization:\n%s" % str(mean_prediction)
            )
            logging.debug(
                "Variance/Uncertainty of network predictions "
                "after unnormalization:\n%s" % str(variance_prediction)
            )

            for i in range(len(network_outputs)):
                network_outputs[i] = zero_mean_unit_var_unnormalization(
                    network_outputs[i], self.y_mean, self.y_std
                )

        if return_individual_predictions:
            return mean_prediction, variance_prediction, network_outputs
        return mean_prediction, variance_prediction
