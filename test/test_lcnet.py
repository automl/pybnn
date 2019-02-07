import unittest

import numpy as np

from pybnn.lcnet import LCNet


class TestLCNet(unittest.TestCase):

    def test_train_predict(self):

        def toy_example(t, a, b):
            return (10 + a * np.log(b * t)) / 10. + 10e-3 * np.random.rand()

        observed = 20
        N = 5
        n_epochs = 10
        observed_t = int(n_epochs * (observed / 100.))

        t_idx = np.arange(1, observed_t + 1) / n_epochs
        t_grid = np.arange(1, n_epochs + 1) / n_epochs

        configs = np.random.rand(N, 2)
        learning_curves = [toy_example(t_grid, configs[i, 0], configs[i, 1]) for i in range(N)]

        X_train = None
        y_train = None
        X_test = None
        y_test = None

        for i in range(N):

            x = np.repeat(configs[i, None, :], t_idx.shape[0], axis=0)
            x = np.concatenate((x, t_idx[:, None]), axis=1)

            x_test = np.concatenate((configs[i, None, :], np.array([[1]])), axis=1)

            lc = learning_curves[i][:observed_t]
            lc_test = np.array([learning_curves[i][-1]])

            if X_train is None:
                X_train = x
                y_train = lc
                X_test = x_test
                y_test = lc_test
            else:
                X_train = np.concatenate((X_train, x), 0)
                y_train = np.concatenate((y_train, lc), 0)
                X_test = np.concatenate((X_test, x_test), 0)
                y_test = np.concatenate((y_test, lc_test), 0)

        print(X_train.shape)
        model = LCNet()

        model.train(X_train, y_train, num_steps=500, num_burn_in_steps=40, lr=1e-2)

        m, v = model.predict(X_test)

        assert len(m.shape) == 1
        assert m.shape[0] == X_test.shape[0]
        assert len(v.shape) == 1
        assert v.shape[0] == X_test.shape[0]


if __name__ == "__main__":
    unittest.main()
