import unittest

import numpy as np

from pybnn.multi_task_bohamiann import MultiTaskBohamiann


class TestMTBohamiann(unittest.TestCase):

    def test_train_predict(self):

        def objective(x, task):
            if task == 0:
                y = 0.5 * np.sin(3 * x[0]) * 4 * (x[0] - 1) * (x[0])
            elif task == 1:
                y = 0.5 * np.sin(3 * x[0] + 1) * 4 * (x[0] - 1) * (x[0])
            elif task == 2:
                y = 0.5 * np.sin(3 * x[0] + 2) * 4 * (x[0] - 1) * (x[0])
            return y

        upper = np.ones(1) * 6

        X = np.random.rand(30, 1) * upper
        y_t0 = np.array([objective(xi, 0) for xi in X[:10]])
        y_t1 = np.array([objective(xi, 1) for xi in X[10:20]])
        y_t2 = np.array([objective(xi, 2) for xi in X[20:]])
        y = np.hstack((y_t0, y_t1, y_t2))

        t_idx = np.zeros([30])
        t_idx[10:20] = 1
        t_idx[20:] = 2

        X = np.append(X, t_idx[:, None], axis=1)

        model = MultiTaskBohamiann(n_tasks=3)

        model.train(X, y, num_steps=500, num_burn_in_steps=40, lr=1e-2)

        X_test = np.random.rand(5, 1) * upper
        X_test = np.append(X_test, np.ones([X_test.shape[0], 1]), axis=1)
        m, v = model.predict(X_test)

        assert len(m.shape) == 1
        assert m.shape[0] == X_test.shape[0]
        assert len(v.shape) == 1
        assert v.shape[0] == X_test.shape[0]


if __name__ == "__main__":
    unittest.main()
