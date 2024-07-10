# %%
import numpy as np
from bpca import BPCA
# from bpca_pymc import BPCA
import matplotlib.pyplot as plt

np.random.seed(123)  # For reproducibility


# hinton diagram
def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    matrix_t = matrix.T

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix_t).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix_t):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


def simulate_data(psi=1, N=100, P=10):
    psi_inv = 1 / psi
    cov = np.diag([5, 4, 3, 2] + [psi_inv] * (P - 4))
    return np.random.multivariate_normal(np.zeros(P), cov, N)


# %%
from scipy.stats import multivariate_normal as mvn, gamma

class GibbsBayesianPCA:
    def __init__(
        self, t, q, a_alpha=1e-3, b_alpha=1e-3, a_tau=1e-3, b_tau=1e-3, beta=1e-3,
        tau_init=None, 
        ):
        self.t = t
        self.N, self.d = t.shape
        self.q = q
        self.a_alpha = a_alpha
        self.b_alpha = b_alpha
        self.a_tau = a_tau
        self.b_tau = b_tau
        self.beta = beta
        self.initialize_parameters(tau_init=tau_init)

    def initialize_parameters(
        self,
        tau_init=None,
        ):
        self.x = np.random.randn(self.N, self.q)
        self.mu = np.zeros((1, self.d))
        self.W = np.random.randn(self.d, self.q)
        self.alpha = gamma.rvs(self.a_alpha, scale=1/self.b_alpha, size=self.q)
        # self.tau = gamma.rvs(self.a_tau, scale=1/self.b_tau)
        # HACK
        # b_tau_tilde= np.abs(np.random.randn(1))
        # a_tau_tilde = self.a_tau + 0.5 * self.N * self.d
        # self.tau = gamma.rvs(a_tau_tilde, scale=1/b_tau_tilde)
        self.tau = 1 if tau_init is None else tau_init
        # b_tau_tilde = self.b_tau_tilde * self.tau

        self.Iq = np.eye(self.q)
        self.Id = np.eye(self.d)


    def update_x(self):
        Sigma_x = np.linalg.inv(self.Iq + self.tau * self.W.T @ self.W)
        for n in range(self.N):
            m_x_n = self.tau * Sigma_x @ self.W.T @ (self.t[[n]] - self.mu).T
            self.x[[n]] = mvn.rvs(mean=m_x_n.flatten(), cov=Sigma_x)

    def update_mu(self):
        Sigma_mu = np.linalg.inv(self.beta * self.Id + self.N * self.tau * self.Id)
        m_mu = self.tau * Sigma_mu @ np.sum(self.t - self.x @ self.W.T, axis=0)
        self.mu = mvn.rvs(mean=m_mu, cov=Sigma_mu)[None, :]

    def update_W(self):
        for k in range(self.d):
            Sigma_w = np.linalg.inv(np.diag(self.alpha) + self.tau * self.x.T @ self.x)
            m_w_k = self.tau * Sigma_w @ self.x.T @ (self.t[:, [k]] - self.mu[:,[k]])
            self.W[k] = mvn.rvs(mean=m_w_k.flatten(), cov=Sigma_w)

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
            np.linalg.norm(self.t[n])**2 + self.mu.T @ self.mu + np.trace(self.W.T @ self.W @ (self.x[n][:,None] @ self.x[n][:,None].T)) + 2 * self.mu @ self.W @ self.x[n] - 2 * self.t[[n]] @ self.W @ self.x[[n]].T - 2 * self.t[[n]] @ self.mu.T
            for n in range(self.N)
        ])
        # FIXME
        # self.tau = gamma.rvs(a_tau_tilde, scale=1/b_tau_tilde)

    def fit(
        self, 
        # params for the gibbs sampler
        iterations=500,
        burn_in=200,
        thinning=10,
        ):

        # store the samples
        self.samples = {
            'x': [],
            'mu': [],
            'W': [],
            'alpha': [],
            'tau': [],
        }

        for i in range(iterations):
            self.update_x()
            self.update_mu()
            self.update_W()
            self.update_alpha()
            self.update_tau()
            if (i + 1) % 100 == 0:
                print('Iteration ', ( i + 1 ))

            if i >= burn_in and i % thinning == 0:
                self.samples['x'].append(self.x)
                self.samples['mu'].append(self.mu)
                self.samples['W'].append(self.W)
                self.samples['alpha'].append(self.alpha)
                self.samples['tau'].append(self.tau)


# %%
# Simulate data with noise
var_noise = 1e-1
data = simulate_data(
    psi=var_noise**(-1), 
    # N=1000
    )



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
# hintion diagram
plt.figure(figsize=(10, 5))
plt.subplot(121)
hinton(bpca_model.mean_w)
plt.title('BPCA (VI)')

# %%
bpca = GibbsBayesianPCA(data, q=data.shape[1]-1, tau_init=var_noise**(-1))



# %%
bpca.fit(iterations=1000)



# %%
bpca_alpha_mean = np.mean(bpca.samples['alpha'][:10] , axis=0)
bpca_tau_mean = np.mean(bpca.samples['tau'])

print("Variance of noise: ", bpca_tau_mean**(-1))
print("Variance of BPCA:\n", bpca_alpha_mean**(-1))


plt.plot(np.log10(sorted(bpca_alpha_mean**(-1), reverse=True)))
plt.xlabel('Principal component')
plt.ylabel('Log10 variance')
plt.title('Estimated noise variance: {:.2e}. Truth {:.2e}'.format(bpca_tau_mean, var_noise) )


bpca.alpha.shape
# %%
sorted(bpca_alpha_mean**(-1))
# %%



# %%
bpca_W_mean = np.mean(bpca.samples['W'], axis=0)
# hintion diagram
plt.figure(figsize=(10, 5))
plt.subplot(121)
hinton(bpca_W_mean)
plt.title('BPCA (Gibbs)')



# %%
bpca.samples['W'].shape
# %%
bpca_W_mean.shape
# %%
