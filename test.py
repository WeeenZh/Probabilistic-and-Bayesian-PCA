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
from scipy.stats import multivariate_normal as mvn, gamma

class GibbsBayesianPCA:
    def __init__(self, t, q, a_alpha=1e-3, b_alpha=1e-3, a_tau=1e-3, b_tau=1e-3, beta=1e-3):
        self.t = t
        self.N, self.d = t.shape
        self.q = q
        self.a_alpha = a_alpha
        self.b_alpha = b_alpha
        self.a_tau = a_tau
        self.b_tau = b_tau
        self.beta = beta
        self.initialize_parameters()

    def initialize_parameters(self):
        self.x = np.random.randn(self.N, self.q)
        self.mu = np.zeros(self.d)
        self.W = np.random.randn(self.d, self.q)
        self.alpha = gamma.rvs(self.a_alpha, scale=1/self.b_alpha, size=self.q)
        # self.tau = gamma.rvs(self.a_tau, scale=1/self.b_tau)
        # HACK
        # b_tau_tilde= np.abs(np.random.randn(1))
        # a_tau_tilde = self.a_tau + 0.5 * self.N * self.d
        # self.tau = gamma.rvs(a_tau_tilde, scale=1/b_tau_tilde)
        self.tau = 1
        # b_tau_tilde = self.b_tau_tilde * self.tau

        self.Iq = np.eye(self.q)
        self.Id = np.eye(self.d)


    def update_x(self):
        Sigma_x = np.linalg.inv(self.Iq + self.tau * self.W.T @ self.W)
        for n in range(self.N):
            m_x_n = self.tau * Sigma_x @ self.W.T @ (self.t[n] - self.mu)
            self.x[n] = mvn.rvs(mean=m_x_n, cov=Sigma_x)

    def update_mu(self):
        Sigma_mu = np.linalg.inv(self.beta * self.Id + self.N * self.tau * self.Id)
        m_mu = self.tau * Sigma_mu @ np.sum(self.t - self.x @ self.W.T, axis=0)
        self.mu = mvn.rvs(mean=m_mu, cov=Sigma_mu)

    def update_W(self):
        for k in range(self.d):
            Sigma_w = np.linalg.inv(np.diag(self.alpha) + self.tau * self.x.T @ self.x)
            m_w_k = self.tau * Sigma_w @ self.x.T @ (self.t[:, k] - self.mu[k])
            self.W[k] = mvn.rvs(mean=m_w_k, cov=Sigma_w)

    def update_alpha(self):
        a_alpha_tilde = self.a_alpha + 0.5 * self.d
        for i in range(self.q):
            b_alpha_tilde = self.b_alpha + 0.5 * np.sum(self.W[:, i] ** 2)
            self.alpha[i] = gamma.rvs(a_alpha_tilde, scale=1/b_alpha_tilde)

    def update_tau(self):
        a_tau_tilde = self.a_tau + 0.5 * self.N * self.d
        # print(self.t[0].shape, self.mu.shape, self.W.shape, self.x[0].shape, self.tau)
        # n = 0
        # print( (self.x[n][:,None] @ self.x[n][:,None].T) )
        # print( np.trace(self.W.T @ self.W @ (self.x[n][:,None] @ self.x[n][:,None].T)) )
        b_tau_tilde = self.b_tau + 0.5 * np.sum([
            np.linalg.norm(self.t[n])**2 + self.mu.T @ self.mu + np.trace(self.W.T @ self.W @ (self.x[n][:,None] @ self.x[n][:,None].T)) + 2 * self.mu @ self.W @ self.x[n] - 2 * self.t[n] @ self.W @ self.x[n] - 2 * self.t[n] @ self.mu
            for n in range(self.N)
        ])
        self.tau = gamma.rvs(a_tau_tilde, scale=1/b_tau_tilde)

    def fit(self, iterations=1000):
        for i in range(iterations):
            self.update_x()
            self.update_mu()
            self.update_W()
            self.update_alpha()
            self.update_tau()
            if (i + 1) % 100 == 0:
                print('Iteration ', ( i + 1 ))


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
