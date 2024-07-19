# %%
import numpy as np
from bpca import BPCA
# from bpca_pymc import BPCA
import matplotlib.pyplot as plt

np.random.seed(0)  # For reproducibility


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

    # # Generate a random orthogonal matrix U
    # random_matrix = np.random.randn(P, P)
    # U, _, _ = np.linalg.svd(random_matrix)
    
    # Transform the covariance matrix
    # cov = U @ cov @ U.T

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
        m_x = self.tau * Sigma_x @ self.W.T @ (self.t - self.mu).T
        for n in range(self.N):
            # m_x_n = self.tau * Sigma_x @ self.W.T @ (self.t[[n]] - self.mu).T
            m_x_n = m_x[:, [n]]
            self.x[[n]] = mvn.rvs(mean=m_x_n.flatten(), cov=Sigma_x)

    def update_mu(self):
        Sigma_mu = np.linalg.inv(self.beta * self.Id + self.N * self.tau * self.Id)
        m_mu = self.tau * Sigma_mu @ np.sum(self.t - self.x @ self.W.T, axis=0)
        self.mu = mvn.rvs(mean=m_mu, cov=Sigma_mu)[None, :]

    def update_W(self):
        Sigma_w = np.linalg.inv(np.diag(self.alpha) + self.tau * self.x.T @ self.x)
        m_w = self.tau * Sigma_w @ self.x.T @ (self.t - self.mu)
        for k in range(self.d):
            # m_w_k = self.tau * Sigma_w @ self.x.T @ (self.t[:, [k]] - self.mu[:,[k]])
            m_w_k = m_w[:, [k]]
            self.W[k] = mvn.rvs(mean=m_w_k.flatten(), cov=Sigma_w)

    def update_alpha(self):
        a_alpha_tilde = self.a_alpha + 0.5 * self.d
        b_alpha_tilde = self.b_alpha + 0.5 * np.sum(self.W ** 2, axis=0)
        for i in range(self.q):
            # b_alpha_tilde_i = self.b_alpha + 0.5 * np.sum(self.W[:, i] ** 2)
            b_alpha_tilde_i = b_alpha_tilde[i]
            self.alpha[i] = gamma.rvs(a_alpha_tilde, scale=1/b_alpha_tilde_i)

    def update_tau(
        self,
        fix_tau_at=None,
        ):
        a_tau_tilde = self.a_tau + 0.5 * self.N * self.d
        # print(self.t[0].shape, self.mu.shape, self.W.shape, self.x[0].shape, self.tau)
        # n = 0
        # print( (self.x[n][:,None] @ self.x[n][:,None].T) )
        # print( np.trace(self.W.T @ self.W @ (self.x[n][:,None] @ self.x[n][:,None].T)) )
        # b_tau_tilde_ = self.b_tau + 0.5 * np.sum([
        #     np.linalg.norm(self.t[n])**2 + self.mu.T @ self.mu + np.trace(self.W.T @ self.W @ (self.x[n][:,None] @ self.x[n][:,None].T)) + 2 * self.mu @ self.W @ self.x[n] - 2 * self.t[[n]] @ self.W @ self.x[[n]].T - 2 * self.t[[n]] @ self.mu.T
        #     for n in range(self.N)
        # ])
        if fix_tau_at is None:
            
            b_tau_tilde = self.b_tau + 0.5 * (
                (self.t**2).sum() + (self.mu**2).sum()*self.N + np.trace(
                    self.W.T @ self.W @ (self.x.T @ self.x)
                ) + 2 * (self.mu @ self.W @ self.x.T).sum() + np.sum([
                    - 2 * self.t[[n]] @ self.W @ self.x[[n]].T 
                    for n in range(self.N)
                ]) - 2 * (self.t @ self.mu.T).sum()
            )
            self.tau = gamma.rvs(a_tau_tilde, scale=1/b_tau_tilde)
        else:
            self.tau = fix_tau_at
            b_tau_tilde = a_tau_tilde / self.tau


    def fit(
        self, 
        # params for the gibbs sampler
        iterations=500,
        burn_in=200,
        thinning=10,
        threshold_alpha_complete=None,
        true_signal_dim=None,
        fix_tau_at=None,
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
            self.iter_converge = iterations
            self.update_x()
            self.update_mu()
            self.update_W()
            self.update_alpha()
            self.update_tau( fix_tau_at=fix_tau_at )
            if (i + 1) % 100 == 0:
                print('Iteration ', ( i + 1 ))

            if i >= burn_in and i % thinning == 0:
                self.samples['x'].append(self.x)
                self.samples['mu'].append(self.mu)
                self.samples['W'].append(self.W)
                self.samples['alpha'].append(self.alpha)
                self.samples['tau'].append(self.tau)

                if (threshold_alpha_complete is not None) and ( true_signal_dim is not None ):
                    # mean of self.samples['alpha']
                    alpha_sorted = sorted(np.mean(self.samples['alpha'] , axis=0))
                    if (alpha_sorted[true_signal_dim] / alpha_sorted[true_signal_dim-1]) > threshold_alpha_complete:
                        self.iter_converge = i
                        break


# %%
var_noise_list = np.logspace(-5, 1, 30)
threshold_alpha_complete = 1e2
iter_end_list = np.zeros(len(var_noise_list))
# variational inference
# n_iter_max = 200000
# gibbs
n_iter_max = 20000
n_repeat = 1
# %%
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import time

# Function to perform the simulation and fitting
def simulate_and_fit(v, n_repeat, n_iter_max, threshold_alpha_complete):
    iter_end_list_i = np.zeros(n_repeat)
    time_list_i = np.zeros(n_repeat)
    for j in range(n_repeat):
        start_time = time.time()
        d = simulate_data(psi=v**(-1), N=100)
        # variational inference
        bpca = BPCA(a_alpha=1e-3, b_alpha=1e-3, a_tau=1e-3, b_tau=1e-3, beta=1e-3)
        bpca.fit(
            d, iters=n_iter_max,
            threshold_alpha_complete=threshold_alpha_complete,
            true_signal_dim=4,
            fix_tau_at=v**(-1),
        )
        # gibbs
        bpca = GibbsBayesianPCA(d, q=d.shape[1]-1, tau_init=v**(-1))
        bpca.fit(
            iterations=n_iter_max,
            threshold_alpha_complete=threshold_alpha_complete,
            true_signal_dim=4,
            fix_tau_at=v**(-1),
            )
        iter_end_list_i[j] = bpca.iter_converge
        time_list_i[j] = time.time() - start_time
    return np.mean(iter_end_list_i), np.mean(time_list_i)


# Specify the number of worker processes (e.g., 4)
num_workers = 30

r_list = []
# Use ProcessPoolExecutor to parallelize the loop
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = {executor.submit(simulate_and_fit, v, n_repeat, n_iter_max, threshold_alpha_complete): v for v in var_noise_list}
    for future in as_completed(futures):
        v = futures[future]
        mean_iter, mean_time = future.result()
        r_list.append((v, mean_iter, mean_time))
        print('Completed Variance: ', v)

# sort r_list by variance
r_list = sorted(r_list, key=lambda x: x[0])
v_list, iter_end_list, time_list = zip(*r_list)
# Convert lists to numpy arrays for plotting
iter_end_list = np.array(iter_end_list)
time_list = np.array(time_list)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot the number of iterations to converge
ax1.plot(np.log10(v_list), iter_end_list, label='Iterations to complete', marker='2')
ax1.axhline(y=n_iter_max, color='r', linestyle='--', label='Maximum iterations', marker='2')
ax1.set_xlabel('Log10 noise variance')
ax1.set_ylabel('Iterations to complete')
ax1.set_title('Iterations to complete vs. noise variance')
ax1.legend()

# Plot the time to converge
ax2.plot(np.log10(v_list), time_list, label='Time to complete', marker='2')
ax2.set_xlabel('Log10 noise variance')
ax2.set_ylabel('Time to complete (seconds)')
ax2.set_title('Time to complete vs. noise variance')
ax2.legend()

# Show the plots
plt.tight_layout()
plt.show()

# Show results excluding those reaching the maximum number of iterations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot the number of iterations to converge excluding max iterations
ax1.plot(np.log10(v_list)[iter_end_list < n_iter_max], iter_end_list[iter_end_list < n_iter_max], marker='2')
ax1.set_xlabel('Log10 noise variance')
ax1.set_ylabel('Iterations to complete')
ax1.set_title('Iterations to complete vs. noise variance (excluding maximum iterations)')

# Plot the time to converge excluding max iterations
ax2.plot(np.log10(v_list)[iter_end_list < n_iter_max], time_list[iter_end_list < n_iter_max], marker='2')
ax2.set_xlabel('Log10 noise variance')
ax2.set_ylabel('Time to complete (seconds)')
ax2.set_title('Time to complete vs. noise variance (excluding maximum iterations)')

# Show the plots
plt.tight_layout()
plt.show()


# %%
# %%
# %%
# %%
# %%





# Simulate data with noise
var_noise = 10
data = simulate_data(
    psi=var_noise**(-1), 
    N=100
    )



# %%
# when there is a gap of threshold_alpha_complete between the eigenvalues of the covariance matrix, it is regarded as converged
threshold_alpha_complete = 1e2
# threshold_alpha_complete = None

# %%
# Initialize and fit BPCA model with variational inference
bpca_model = BPCA(a_alpha=1e-3, b_alpha=1e-3, a_tau=1e-3, b_tau=1e-3, beta=1e-3)
bpca_model.fit(
    data, iters=100000,
    threshold_alpha_complete=threshold_alpha_complete,
    true_signal_dim=4,
    )
print("Converged in ", bpca_model.iter_converge, " iterations")

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
bpca.fit(
    iterations=1000,
    threshold_alpha_complete=threshold_alpha_complete,
    true_signal_dim=4,
    )
print("Converged in ", bpca.iter_converge, " iterations")


# %%
%%timeit -r 3 -n 10
bpca.fit(iterations=10)




# %%
bpca_alpha_mean = np.mean(bpca.samples['alpha'] , axis=0)
bpca_tau_mean = np.mean(bpca.samples['tau'])

print("Variance of noise: ", bpca_tau_mean**(-1))
print("Variance of BPCA:\n", bpca_alpha_mean**(-1))


plt.plot(np.log10(sorted(bpca_alpha_mean**(-1), reverse=True)))
plt.xlabel('Principal component')
plt.ylabel('Log10 variance')
plt.title('Estimated noise variance: {:.2e}. Truth {:.2e}'.format(bpca_tau_mean**(-1), var_noise) )



# %%
bpca_W_mean = np.mean(bpca.samples['W'], axis=0)
# hintion diagram
plt.figure(figsize=(10, 5))
plt.subplot(121)
hinton(bpca_W_mean)
plt.title('BPCA (Gibbs)')



# %%
