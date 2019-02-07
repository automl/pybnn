import unittest

import numpy as np
from pybnn.dngo import DNGO


class TestDNGO(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 3)
        self.y = np.sinc(self.X * 10 - 5).sum(axis=1)

    def test_mcmc(self):
        model = DNGO(num_epochs=10, burnin_steps=10, chain_length=20, do_mcmc=True)
        model.train(self.X, self.y)

        X_test = np.random.rand(10, self.X.shape[1])

        m, v = model.predict(X_test)

        assert len(m.shape) == 1
        assert m.shape[0] == X_test.shape[0]
        assert len(v.shape) == 1
        assert v.shape[0] == X_test.shape[0]

    def test_ml(self):
        model = DNGO(num_epochs=10, do_mcmc=False)
        model.train(self.X, self.y)

        X_test = np.random.rand(10, self.X.shape[1])

        m, v = model.predict(X_test)

        assert len(m.shape) == 1
        assert m.shape[0] == X_test.shape[0]
        assert len(v.shape) == 1
        assert v.shape[0] == X_test.shape[0]

    def test_without_normalization(self):
        model = DNGO(num_epochs=10, do_mcmc=False, normalize_output=False, normalize_input=False)
        model.train(self.X, self.y)

        X_test = np.random.rand(10, self.X.shape[1])

        m, v = model.predict(X_test)

        assert len(m.shape) == 1
        assert m.shape[0] == X_test.shape[0]
        assert len(v.shape) == 1
        assert v.shape[0] == X_test.shape[0]

    def test_incumbent(self):
        model = DNGO(num_epochs=10, do_mcmc=False)
        model.train(self.X, self.y)

        x_star, y_star = model.get_incumbent()

        b = np.argmin(self.y)

        assert np.all(np.isclose(x_star, self.X[b]))
        assert np.all(np.isclose(y_star, self.y[b]))


if __name__ == "__main__":
    unittest.main()
