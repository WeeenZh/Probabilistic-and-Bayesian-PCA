# %%
import numpy as np
import pymc
import matplotlib.pyplot as plt

class BayesianPCA:
    def __init__(self, t, q, a_alpha=1e-3, b_alpha=1e-3, a_tau=1e-3, b_tau=1e-3, beta=1e-3):
        self.t = t
        self.N, self.d = t.shape
        self.q = q
        self.a_alpha = a_alpha
        self.b_alpha = b_alpha
        self.a_tau = a_tau
        self.b_tau = b_tau
        self.beta = beta
        self.model = self.create_model()

    def create_model(self):
        t = self.t
        N, d = self.N, self.d
        q = self.q
        a_alpha, b_alpha = self.a_alpha, self.b_alpha
        a_tau, b_tau = self.a_tau, self.b_tau
        beta = self.beta

        # Priors
        mu = pymc.Normal('mu', mu=0, tau=beta, size=d)
        W = pymc.Normal('W', mu=0, tau=1.0, size=(d, q))
        alpha = pymc.Gamma('alpha', alpha=a_alpha, beta=b_alpha, size=q)
        tau = pymc.Gamma('tau', alpha=a_tau, beta=b_tau)

        # Latent variables
        x = pymc.Normal('x', mu=0, tau=1.0, size=(N, q))

        @pymc.deterministic
        def t_pred(mu=mu, W=W, x=x):
            return np.dot(x, W.T) + mu

        # Likelihood
        t_obs = pymc.Normal('t_obs', mu=t_pred, tau=tau, observed=True, value=t)

        return locals()

    def fit(self, iterations=10000, burn=5000, thin=10, sampler='Metropolis'):
        self.mcmc = pymc.MCMC(self.model)
        
        # Assigning different samplers to variables
        if sampler == 'Metropolis':
            self.mcmc.use_step_method(pymc.Metropolis, self.model['mu'])
            self.mcmc.use_step_method(pymc.Metropolis, self.model['W'])
            self.mcmc.use_step_method(pymc.Metropolis, self.model['alpha'])
            self.mcmc.use_step_method(pymc.Metropolis, self.model['tau'])
            self.mcmc.use_step_method(pymc.Metropolis, self.model['x'])
        elif sampler == 'AdaptiveMetropolis':
            self.mcmc.use_step_method(pymc.AdaptiveMetropolis, self.model['mu'])
            self.mcmc.use_step_method(pymc.AdaptiveMetropolis, self.model['W'])
            self.mcmc.use_step_method(pymc.AdaptiveMetropolis, self.model['alpha'])
            self.mcmc.use_step_method(pymc.AdaptiveMetropolis, self.model['tau'])
            self.mcmc.use_step_method(pymc.AdaptiveMetropolis, self.model['x'])
        elif sampler == 'Slice':
            self.mcmc.use_step_method(pymc.Slice, self.model['mu'])
            self.mcmc.use_step_method(pymc.Slice, self.model['W'])
            self.mcmc.use_step_method(pymc.Slice, self.model['alpha'])
            self.mcmc.use_step_method(pymc.Slice, self.model['tau'])
            self.mcmc.use_step_method(pymc.Slice, self.model['x'])
        elif sampler == 'Gibbs':
            self.mcmc.use_step_method(pymc.Gibbs, self.model['mu'])
            self.mcmc.use_step_method(pymc.Gibbs, self.model['W'])
            self.mcmc.use_step_method(pymc.Gibbs, self.model['alpha'])
            self.mcmc.use_step_method(pymc.Gibbs, self.model['tau'])
            self.mcmc.use_step_method(pymc.Gibbs, self.model['x'])
        
        self.mcmc.sample(iter=iterations, burn=burn, thin=thin, progress_bar=False)

    def get_results(self):
        mu_samples = self.mcmc.trace('mu')[:]
        W_samples = self.mcmc.trace('W')[:]
        alpha_samples = self.mcmc.trace('alpha')[:]
        tau_samples = self.mcmc.trace('tau')[:]
        x_samples = self.mcmc.trace('x')[:]
        return mu_samples, W_samples, alpha_samples, tau_samples, x_samples


# %%
# Example usage
np.random.seed(123)
t = np.random.randn(100, 10)  # Replace with actual data
bpca = BayesianPCA(t, q=5)
bpca.fit(iterations=1000, burn=500, thin=10)
mu_samples, W_samples, alpha_samples, tau_samples, x_samples = bpca.get_results()

# %%

def simulate_data(psi=1, N=100, P=10):
    psi_inv = 1 / psi
    cov = np.diag([5, 4, 3, 2] + [psi_inv] * (P - 4))

    # Generate a random orthogonal matrix U
    random_matrix = np.random.randn(P, P)
    U, _, _ = np.linalg.svd(random_matrix)
    
    # Transform the covariance matrix
    cov = U @ cov @ U.T

    return np.random.multivariate_normal(np.zeros(P), cov, N)

# %%
# %%
# Simulate data with noise
var_noise = 1e-2
data = simulate_data(
    psi=var_noise**(-1), 
    N=100
    )


# %%
bpca = BayesianPCA(data, q=data.shape[1] - 1)
bpca.fit(
    iterations=10000, burn=5000, thin=10, 
    sampler='Metropolis',
    # sampler='AdaptiveMetropolis',
    # sampler='Slice',
    # sampler='Gibbs',
    )



# %%
# alpha_samples
alpha_samples = bpca.mcmc.trace('alpha')[:]
alpha_samples_mean = alpha_samples.mean(axis=0)
plt.plot(sorted(alpha_samples_mean**(-1), reverse=True))


# %%
alpha_samples_mean
# %%
