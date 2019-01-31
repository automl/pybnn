import unittest

import numpy as np
from scipy.optimize import check_grad

from pybnn.bohamiann import Bohamiann


class TestBohamiann(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 3)
        self.y = np.sinc(self.X * 10 - 5).sum(axis=1)
        self.model = Bohamiann(normalize_input=True, normalize_output=True, use_double_precision=True)
        self.model.train(self.X, self.y, num_burn_in_steps=20, num_steps=100, keep_every=10)

    def test_predict(self):
        X_test = np.random.rand(10, self.X.shape[1])

        m, v = self.model.predict(X_test)

        assert len(m.shape) == 1
        assert m.shape[0] == X_test.shape[0]
        assert len(v.shape) == 1
        assert v.shape[0] == X_test.shape[0]

    def test_gradient_mean(self):
        X_test = np.random.rand(10, self.X.shape[1])

        def wrapper(x):
            return self.model.predict([x])[0]

        def wrapper_grad(x):
            return self.model.predictive_mean_gradient(x)

        grad = self.model.predictive_mean_gradient(X_test[0])
        assert grad.shape[0] == X_test.shape[1]

        for xi in X_test:
            err = check_grad(wrapper, wrapper_grad, xi, epsilon=1e-6)
            assert err < 1e-5

    def test_gradient_variance(self):
        X_test = np.random.rand(10, self.X.shape[1])

        def wrapper(x):
            v = self.model.predict([x])[1]
            return v

        def wrapper_grad(x):
            return self.model.predictive_variance_gradient(x)

        grad = self.model.predictive_variance_gradient(X_test[0])
        assert grad.shape[0] == X_test.shape[1]

        for xi in X_test:
            err = check_grad(wrapper, wrapper_grad, xi, epsilon=1e-6)
            assert err < 1e-5


class TestBohamiannSampler(unittest.TestCase):

    def test_sgld(self):
        self.X = np.random.rand(10, 3)
        self.y = np.sinc(self.X * 10 - 5).sum(axis=1)
        self.model = Bohamiann(normalize_input=True, normalize_output=True,
                               use_double_precision=True, sampling_method="sgld")
        self.model.train(self.X, self.y, num_burn_in_steps=20, num_steps=100, keep_every=10)

    def test_preconditioned_sgld(self):
        self.X = np.random.rand(10, 3)
        self.y = np.sinc(self.X * 10 - 5).sum(axis=1)
        self.model = Bohamiann(normalize_input=True, normalize_output=True,
                               use_double_precision=True, sampling_method="preconditioned_sgld")
        self.model.train(self.X, self.y, num_burn_in_steps=20, num_steps=100, keep_every=10)

    def test_sghmc(self):
        self.X = np.random.rand(10, 3)
        self.y = np.sinc(self.X * 10 - 5).sum(axis=1)
        self.model = Bohamiann(normalize_input=True, normalize_output=True,
                               use_double_precision=True, sampling_method="sghmc")
        self.model.train(self.X, self.y, num_burn_in_steps=20, num_steps=100, keep_every=10)

    def test_adaptive_sghmc(self):
        self.X = np.random.rand(10, 3)
        self.y = np.sinc(self.X * 10 - 5).sum(axis=1)
        self.model = Bohamiann(normalize_input=True, normalize_output=True,
                               use_double_precision=True, sampling_method="adaptive_sghmc")
        self.model.train(self.X, self.y, num_burn_in_steps=20, num_steps=100, keep_every=10)


if __name__ == "__main__":
    unittest.main()
