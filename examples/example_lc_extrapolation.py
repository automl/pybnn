import numpy as np
import time
import matplotlib.pyplot as plt
from pybnn.lc_extrapolation.learning_curves import MCMCCurveModelCombination


observed = 40
n_epochs = 100

t_idx = np.arange(1, observed+1)
t_idx_full = np.arange(1, n_epochs+1)


def toy_example(t, a, b):
    return (10 + a * np.log(b * t + 1e-8)) / 10.  # + 10e-3 * np.random.rand()


a = np.random.rand()
b = np.random.rand()
lc = [toy_example(t / n_epochs, a, b) for t in t_idx_full]

model = MCMCCurveModelCombination(n_epochs + 1,
                                  nwalkers=50,
                                  nsamples=800,
                                  burn_in=500,
                                  recency_weighting=False,
                                  soft_monotonicity_constraint=False,
                                  monotonicity_constraint=True,
                                  initial_model_weight_ml_estimate=True)
st = time.time()
model.fit(t_idx, lc[:observed])
print("Training time: %.2f" % (time.time() - st))


st = time.time()
p_greater = model.posterior_prob_x_greater_than(n_epochs + 1, .5)
print("Prediction time: %.2f" % (time.time() - st))

m = np.zeros([n_epochs])
s = np.zeros([n_epochs])

for i in range(n_epochs):
    p = model.predictive_distribution(i+1)
    m[i] = np.mean(p)
    s[i] = np.std(p)

mean_mcmc = m[-1]
std_mcmc = s[-1]

plt.plot(t_idx_full, m, color="purple", label="LC-Extrapolation")
plt.fill_between(t_idx_full, m + s, m - s, alpha=0.2, color="purple")
plt.plot(t_idx_full, lc)

plt.xlim(1, n_epochs)
plt.legend()
plt.xlabel("Number of epochs")
plt.ylabel("Validation error")
plt.axvline(observed, linestyle="--", color="black")
plt.show()