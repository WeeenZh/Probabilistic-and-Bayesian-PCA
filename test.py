# %%
import numpy as np
from bpca import BPCA
# from bpca_pymc import BPCA
import matplotlib.pyplot as plt

np.random.seed(123)  # For reproducibility

def simulate_data(psi=1, N=100, P=10):
    psi_inv = 1 / psi
    cov = np.diag([5, 4, 3, 2] + [psi_inv] * (P - 4))
    return np.random.multivariate_normal(np.zeros(P), cov, N)

# %%
# Simulate data with noise
var_noise = 1e0
data = simulate_data(psi=var_noise**(-1))

# %%
# Initialize and fit BPCA model with variational inference
bpca_model = BPCA(a_alpha=1e-3, b_alpha=1e-3, a_tau=1e-3, b_tau=1e-3, beta=1e-3)
bpca_model.fit(data, iters=10000)

var_noise_est = bpca_model.tau**(-1)
print("Variational of noise: ", var_noise_est)

variance = bpca_model.get_inv_variance() ** (-1)
print("Variance of BPCA:\n", variance)

# plot the sorted variance in log10 scale
plt.plot(np.log10(sorted(variance, reverse=True)))
plt.xlabel('Principal component')
plt.ylabel('Log10 variance')
plt.title('Estimated noise variance: {:.2e}. Truth {:.2e}'.format(var_noise_est, var_noise) )
plt.show()

# %%
from gibs_bpca import GibbsBayesianPCA
# %%
data.shape
# %%
bpca = GibbsBayesianPCA(data, q=data.shape[1])
bpca.fit(iterations=3000)

# %%
bpca.tau, bpca.alpha**(-1)
plt.plot(np.log10(sorted(bpca.alpha**(-1), reverse=True)))
plt.xlabel('Principal component')
plt.ylabel('Log10 variance')
plt.title('Estimated noise variance: {:.2e}. Truth {:.2e}'.format(bpca.tau, var_noise) )


# %%
bpca.alpha.shape
# %%
