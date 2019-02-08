import logging
import time
import typing
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from scipy.stats import norm

from pybnn.base_model import BaseModel
from pybnn.priors import weight_prior, log_variance_prior
from pybnn.sampler import AdaptiveSGHMC, SGLD, SGHMC, PreconditionedSGLD
from pybnn.util.infinite_dataloader import infinite_dataloader
from pybnn.util.layers import AppendLayer
from pybnn.util.normalization import zero_mean_unit_var_denormalization, zero_mean_unit_var_normalization


def get_default_network(input_dimensionality: int) -> torch.nn.Module:
    class Architecture(torch.nn.Module):
        def __init__(self, n_inputs, n_hidden=50):
            super(Architecture, self).__init__()
            self.fc1 = torch.nn.Linear(n_inputs, n_hidden)
            self.fc2 = torch.nn.Linear(n_hidden, n_hidden)
            self.fc3 = torch.nn.Linear(n_hidden, 1)
            self.log_std = AppendLayer(noise=1e-3)

        def forward(self, input):
            x = torch.tanh(self.fc1(input))
            x = torch.tanh(self.fc2(x))
            x = self.fc3(x)
            return self.log_std(x)

    return Architecture(n_inputs=input_dimensionality)


def nll(input: torch.Tensor, target: torch.Tensor):
    """
    computes the average negative log-likelihood (Gaussian)

    :param input: mean and variance predictions of the networks
    :param target: target values
    :return: negative log-likelihood
    """
    batch_size = input.size(0)

    prediction_mean = input[:, 0].view((-1, 1))
    log_prediction_variance = input[:, 1].view((-1, 1))
    prediction_variance_inverse = 1. / (torch.exp(log_prediction_variance) + 1e-16)

    mean_squared_error = (target.view(-1, 1) - prediction_mean) ** 2

    log_likelihood = torch.sum(
        torch.sum(-mean_squared_error * (0.5 * prediction_variance_inverse) - 0.5 * log_prediction_variance, dim=1))

    log_likelihood = log_likelihood / batch_size

    return -log_likelihood


class Bohamiann(BaseModel):
    def __init__(self,
                 get_network=get_default_network,
                 normalize_input: bool = True,
                 normalize_output: bool = True,
                 sampling_method: str = "adaptive_sghmc",
                 use_double_precision: bool = True,
                 metrics=(nn.MSELoss,),
                 likelihood_function=nll,
                 print_every_n_steps=100,
                 ) -> None:
        """

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

        :param get_network: function handle that returns the archtiecture
        :param normalize_input: defines whether to normalize the inputs
        :param normalize_output: defines whether to normalize the outputs
        :param sampling_method: specifies the sampling strategy,
        options: {sgld, sghmc, adaptive_sghmc, preconditioned_sgld}
        :param use_double_precision: defines whether to use double or float precisions
        :param metrics: metrics to evaluate
        :param likelihood_function: function handle that computes the training loss
        :param print_every_n_steps: defines after how many the current loss is printed
        """
        self.print_every_n_steps = print_every_n_steps
        self.metrics = metrics
        self.do_normalize_input = normalize_input
        self.do_normalize_output = normalize_output
        self.get_network = get_network
        self.is_trained = False
        self.use_double_precision = use_double_precision
        self.sampling_method = sampling_method
        self.sampled_weights = []  # type: typing.List[typing.Tuple[np.ndarray]]
        self.likelihood_function = likelihood_function
        self.sampler = None

    @property
    def network_weights(self) -> tuple:
        """
        Extract current network weight values as `np.ndarray`.

        :return: Tuple containing current network weight values
        """
        return tuple(
            np.asarray(torch.tensor(parameter.data).numpy())
            for parameter in self.model.parameters()
        )

    @network_weights.setter
    def network_weights(self, weights: typing.List[np.ndarray]) -> None:
        """
        Assign new `weights` to our neural networks parameters.

        :param weights: List of weight values to assign.
            Individual list elements must have shapes that match
            the network parameters with the same index in `self.network_weights`.
        """
        logging.debug("Assigning new network weights")
        for parameter, sample in zip(self.model.parameters(), weights):
            parameter.copy_(torch.from_numpy(sample))

    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              num_steps: int = 13000,
              keep_every: int = 100,
              num_burn_in_steps: int = 3000,
              lr: float = 1e-5,
              batch_size=20,
              epsilon: float = 1e-10,
              mdecay: float = 0.05,
              continue_training: bool = False,
              verbose: bool = False,
              **kwargs):

        """
        Train a BNN using input datapoints `x_train` with corresponding targets `y_train`.

        :param x_train: input training datapoints.
        :param y_train: input training targets.
        :param num_steps: Number of sampling steps to perform after burn-in is finished.
            In total, `num_steps // keep_every` network weights will be sampled.
        :param keep_every: Number of sampling steps (after burn-in) to perform before keeping a sample.
            In total, `num_steps // keep_every` network weights will be sampled.
        :param num_burn_in_steps: Number of burn-in steps to perform.
            This value is passed to the given `optimizer` if it supports special
            burn-in specific behavior.
            Networks sampled during burn-in are discarded.
        :param lr: learning rate
        :param batch_size: batch size
        :param epsilon: epsilon for numerical stability
        :param mdecay: momemtum decay
        :param continue_training: defines whether we want to continue from the last training run
        :param verbose: verbose output
        """
        logging.debug("Training started.")
        start_time = time.time()

        num_datapoints, input_dimensionality = x_train.shape
        logging.debug(
            "Processing %d training datapoints "
            " with % dimensions each." % (num_datapoints, input_dimensionality)
        )
        assert batch_size >= 1, "Invalid batch size. Batches must contain at least a single sample."
        assert len(y_train.shape) == 1 or (len(y_train.shape) == 2 and y_train.shape[
            1] == 1), "Targets need to be in vector format, i.e (N,) or (N,1)"

        if x_train.shape[0] < batch_size:
            logging.warning("Not enough datapoints to form a batch. Use all datapoints in each batch")
            batch_size = x_train.shape[0]

        self.X = x_train
        if len(y_train.shape) == 2:
            self.y = y_train[:, 0]
        else:
            self.y = y_train

        if self.do_normalize_input:
            logging.debug(
                "Normalizing training datapoints to "
                " zero mean and unit variance."
            )
            x_train_, self.x_mean, self.x_std = self.normalize_input(x_train)
            if self.use_double_precision:
                x_train_ = torch.from_numpy(x_train_).double()
            else:
                x_train_ = torch.from_numpy(x_train_).float()
        else:
            if self.use_double_precision:
                x_train_ = torch.from_numpy(x_train).double()
            else:
                x_train_ = torch.from_numpy(x_train).float()

        if self.do_normalize_output:
            logging.debug("Normalizing training labels to zero mean and unit variance.")
            y_train_, self.y_mean, self.y_std = self.normalize_output(self.y)

            if self.use_double_precision:
                y_train_ = torch.from_numpy(y_train_).double()
            else:
                y_train_ = torch.from_numpy(y_train_).float()
        else:
            if self.use_double_precision:
                y_train_ = torch.from_numpy(y_train).double()
            else:
                y_train_ = torch.from_numpy(y_train).float()

        train_loader = infinite_dataloader(
            data_utils.DataLoader(
                data_utils.TensorDataset(x_train_, y_train_),
                batch_size=batch_size,
                shuffle=True
            )
        )

        if self.use_double_precision:
            dtype = np.float64
        else:
            dtype = np.float32

        if not continue_training:
            logging.debug("Clearing list of sampled weights.")

            self.sampled_weights.clear()
            if self.use_double_precision:
                self.model = self.get_network(input_dimensionality=input_dimensionality).double()
            else:
                self.model = self.get_network(input_dimensionality=input_dimensionality).float()

            if self.sampling_method == "adaptive_sghmc":
                self.sampler = AdaptiveSGHMC(self.model.parameters(),
                                             scale_grad=dtype(num_datapoints),
                                             num_burn_in_steps=num_burn_in_steps,
                                             lr=dtype(lr),
                                             mdecay=dtype(mdecay),
                                             epsilon=dtype(epsilon))
            elif self.sampling_method == "sgld":
                self.sampler = SGLD(self.model.parameters(),
                                    lr=dtype(lr),
                                    scale_grad=num_datapoints)
            elif self.sampling_method == "preconditioned_sgld":
                self.sampler = PreconditionedSGLD(self.model.parameters(),
                                                  lr=dtype(lr),
                                                  num_train_points=num_datapoints)
            elif self.sampling_method == "sghmc":
                self.sampler = SGHMC(self.model.parameters(),
                                     scale_grad=dtype(num_datapoints),
                                     mdecay=dtype(mdecay),
                                     lr=dtype(lr))

        batch_generator = islice(enumerate(train_loader), num_steps)

        for step, (x_batch, y_batch) in batch_generator:
            self.sampler.zero_grad()
            loss = self.likelihood_function(input=self.model(x_batch), target=y_batch)
            # Add prior. Note the gradient is computed by: g_prior + N/n sum_i grad_theta_xi see Eq 4
            # in Welling and Whye The 2011. Because of that we divide here by N=num of datapoints since
            # in the sample we rescale the gradient by N again
            loss -= log_variance_prior(self.model(x_batch)[:, 1].view((-1, 1))) / num_datapoints
            loss -= weight_prior(self.model.parameters(), dtype=dtype) / num_datapoints
            loss.backward()
            self.sampler.step()

            if verbose and step > 0 and step % self.print_every_n_steps == 0:

                # compute the training performance of the ensemble
                if len(self.sampled_weights) > 1:
                    mu, var = self.predict(x_train)
                    total_nll = -np.mean(norm.logpdf(y_train, loc=mu, scale=np.sqrt(var)))
                    total_mse = np.mean((y_train - mu) ** 2)
                # in case we do not have an ensemble we compute the performance of the last weight sample
                else:
                    f = self.model(x_train_)

                    if self.do_normalize_output:
                        mu = zero_mean_unit_var_denormalization(f[:, 0], self.y_mean, self.y_std).data.numpy()
                        var = torch.exp(f[:, 1]) * self.y_std ** 2
                        var = var.data.numpy()
                    else:
                        mu = f[:, 0].data.numpy()
                        var = f[:, 1].data.numpy()
                    total_nll = -np.mean(norm.logpdf(y_train, loc=mu, scale=np.sqrt(var)))
                    total_mse = np.mean((y_train - mu) ** 2)

                t = time.time() - start_time

                if step < num_burn_in_steps:
                    print("Step {:8d} : NLL = {:11.4e} MSE = {:.4e} "
                          "Time = {:5.2f}".format(step, float(total_nll),
                                                  float(total_mse), t))

                if step > num_burn_in_steps:
                    print("Step {:8d} : NLL = {:11.4e} MSE = {:.4e} "
                          "Samples= {} Time = {:5.2f}".format(step,
                                                              float(total_nll),
                                                              float(total_mse),
                                                              len(self.sampled_weights), t))

            if step > num_burn_in_steps and (step - num_burn_in_steps) % keep_every == 0:
                weights = self.network_weights

                self.sampled_weights.append(weights)

        self.is_trained = True

    def train_and_evaluate(self, x_train: np.ndarray, y_train: np.ndarray,
                           x_valid: np.ndarray, y_valid: np.ndarray,
                           num_steps: int = 13000,
                           validate_every_n_steps=1000,
                           keep_every: int = 100,
                           num_burn_in_steps: int = 3000,
                           lr: float = 1e-2,
                           epsilon: float = 1e-10,
                           batch_size: int = 20,
                           mdecay: float = 0.05,
                           verbose=False):
        """
        Train and validates the neural network

        :param x_train: input training datapoints.
        :param y_train: input training targets.
        :param x_valid: validation data points
        :param y_valid: valdiation targets
        :param num_steps:  Number of sampling steps to perform after burn-in is finished.
            In total, `num_steps // keep_every` network weights will be sampled.
        :param validate_every_n_steps:
        :param keep_every: Number of sampling steps (after burn-in) to perform before keeping a sample.
            In total, `num_steps // keep_every` network weights will be sampled.
        :param num_burn_in_steps: Number of burn-in steps to perform.
            This value is passed to the given `optimizer` if it supports special
            burn-in specific behavior.
            Networks sampled during burn-in are discarded.
        :param lr: learning rate
        :param batch_size: batch size
        :param epsilon: epsilon for numerical stability
        :param mdecay: momemtum decay
        :param verbose: verbose output

        """
        assert batch_size >= 1, "Invalid batch size. Batches must contain at least a single sample."

        if x_train.shape[0] < batch_size:
            logging.warning("Not enough datapoints to form a batch. Use all datapoints in each batch")
            batch_size = x_train.shape[0]

        # burn-in
        self.train(x_train, y_train, num_burn_in_steps=num_burn_in_steps, num_steps=num_burn_in_steps,
                   lr=lr, epsilon=epsilon, mdecay=mdecay, verbose=verbose)

        learning_curve_mse = []
        learning_curve_ll = []
        n_steps = []
        for i in range(num_steps // validate_every_n_steps):
            self.train(x_train, y_train, num_burn_in_steps=0, num_steps=validate_every_n_steps,
                       lr=lr, epsilon=epsilon, mdecay=mdecay, verbose=verbose, keep_every=keep_every,
                       continue_training=True, batch_size=batch_size)

            mu, var = self.predict(x_valid)

            ll = np.mean(norm.logpdf(y_valid, loc=mu, scale=np.sqrt(var)))
            mse = np.mean((y_valid - mu) ** 2)
            step = num_burn_in_steps + (i + 1) * validate_every_n_steps

            learning_curve_ll.append(ll)
            learning_curve_mse.append(mse)
            n_steps.append(step)

            if verbose:
                print("Validate : NLL = {:11.4e} MSE = {:.4e}".format(-ll, mse))

        return n_steps, learning_curve_ll, learning_curve_mse

    def normalize_input(self, x, m=None, s=None):
        """
        Normalizes input

        :param x: data
        :param m: mean
        :param s: standard deviation
        :return: normalized input
        """

        return zero_mean_unit_var_normalization(x, m, s)

    def normalize_output(self, x, m=None, s=None):
        """
        Normalizes output

        :param x: targets
        :param m: mean
        :param s: standard deviation
        :return: normalized targets
        """
        return zero_mean_unit_var_normalization(x, m, s)

    def predict(self, x_test: np.ndarray, return_individual_predictions: bool = False):
        """
        Predicts mean and variance for the given test point

        :param x_test: test datapoint
        :param return_individual_predictions: if True also the predictions of the individual models are returned
        :return: mean and variance
        """
        x_test_ = np.asarray(x_test)

        if self.do_normalize_input:
            x_test_, *_ = self.normalize_input(x_test_, self.x_mean, self.x_std)

        def network_predict(x_test_, weights):
            with torch.no_grad():
                self.network_weights = weights
                if self.use_double_precision:
                    return self.model(torch.from_numpy(x_test_).double()).numpy()
                else:
                    return self.model(torch.from_numpy(x_test_).float()).numpy()

        logging.debug("Predicting with %d networks." % len(self.sampled_weights))
        network_outputs = np.array([
            network_predict(x_test_, weights=weights)
            for weights in self.sampled_weights
        ])

        mean_prediction = np.mean(network_outputs[:, :, 0], axis=0)
        # variance_prediction = np.mean((network_outputs[:, :, 0] - mean_prediction) ** 2, axis=0)
        # Total variance
        variance_prediction = np.mean((network_outputs[:, :, 0] - mean_prediction) ** 2
                                      + np.exp(network_outputs[:, :, 1]), axis=0)

        if self.do_normalize_output:

            mean_prediction = zero_mean_unit_var_denormalization(
                mean_prediction, self.y_mean, self.y_std
            )
            variance_prediction *= self.y_std ** 2

            for i in range(len(network_outputs)):
                network_outputs[i] = zero_mean_unit_var_denormalization(
                    network_outputs[i], self.y_mean, self.y_std
                )

        if return_individual_predictions:
            return mean_prediction, variance_prediction, network_outputs[:, :, 0]

        return mean_prediction, variance_prediction

    def predict_single(self, x_test: np.ndarray, sample_index: int):
        """
        Compute the prediction of a single weight sample

        :param x_test: test datapoint
        :param sample_index: specifies the index of the weight sample
        :return: mean and variance of the neural network
        """
        x_test_ = np.asarray(x_test)

        if self.do_normalize_input:
            x_test_, *_ = self.normalize_input(x_test_, self.x_mean, self.x_std)

        def network_predict(x_test_, weights):
            with torch.no_grad():
                self.network_weights = weights
                if self.use_double_precision:
                    return self.model(torch.from_numpy(x_test_).double()).numpy()
                else:
                    return self.model(torch.from_numpy(x_test_).float()).numpy()

        logging.debug("Predicting with %d networks." % len(self.sampled_weights))
        function_value = np.array(network_predict(x_test_, weights=self.sampled_weights[sample_index]))

        if self.do_normalize_output:
            function_value = zero_mean_unit_var_denormalization(
                function_value, self.y_mean, self.y_std
            )
        return function_value

    def f_gradient(self, x_test, weights):
        x_test_ = np.asarray(x_test)

        with torch.no_grad():
            self.network_weights = weights

        if self.use_double_precision:
            x = torch.autograd.Variable(torch.from_numpy(x_test_[None, :]).double(), requires_grad=True)
        else:
            x = torch.autograd.Variable(torch.from_numpy(x_test_[None, :]).float(), requires_grad=True)

        if self.do_normalize_input:
            if self.use_double_precision:
                x_mean = torch.autograd.Variable(torch.from_numpy(self.x_mean).double(), requires_grad=False)
                x_std = torch.autograd.Variable(torch.from_numpy(self.x_std).double(), requires_grad=False)
            else:
                x_mean = torch.autograd.Variable(torch.from_numpy(self.x_mean).float(), requires_grad=False)
                x_std = torch.autograd.Variable(torch.from_numpy(self.x_std).float(), requires_grad=False)

            x_norm = (x - x_mean) / x_std
            m = self.model(x_norm)[0][0]
        else:
            m = self.model(x)[0][0]
        if self.do_normalize_output:

            if self.use_double_precision:
                y_mean = torch.autograd.Variable(torch.from_numpy(np.array([self.y_mean])).double(),
                                                 requires_grad=False)
                y_std = torch.autograd.Variable(torch.from_numpy(np.array([self.y_std])).double(), requires_grad=False)

            else:
                y_mean = torch.autograd.Variable(torch.from_numpy(np.array([self.y_mean])).float(), requires_grad=False)
                y_std = torch.autograd.Variable(torch.from_numpy(np.array([self.y_std])).float(), requires_grad=False)

            m = m * y_std + y_mean

        m.backward()

        g = x.grad.data.numpy()[0, :]
        return g

    def predictive_mean_gradient(self, x_test: np.ndarray):

        # compute the individual gradients for each weight vector
        grads = np.array([self.f_gradient(x_test, weights=weights) for weights in self.sampled_weights])

        # the gradient of the mean is mean of all individual gradients
        g = np.mean(grads, axis=0)

        return g

    def predictive_variance_gradient(self, x_test: np.ndarray):
        m, v, funcs = self.predict(x_test[None, :], return_individual_predictions=True)

        grads = np.array([self.f_gradient(x_test, weights=weights) for weights in self.sampled_weights])

        dmdx = self.predictive_mean_gradient(x_test)

        g = np.mean([2 * (funcs[i] - m) * (grads[i] - dmdx) for i in range(len(self.sampled_weights))], axis=0)

        return g
