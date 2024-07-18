library(MASS)
library(R2jags)
library(bPCA)
library(Matrix)
library(coda)

eigenvalplotslg.bPCA <- function(bPCA.fitted, data)  
{
    V <- ncol(data)
    sims <- bPCA.fitted$BUGSoutput$sims.matrix
    # extracting only the covariances
    sims <- sims[,1:(V*V)]
    # empty matrix for results
    eigen.chains <- matrix(nrow=nrow(sims), ncol=V)
    
    # calculate eigenvalues for each covariance matrix in the chain
    for(i in 1:nrow(sims))
    {
      covm <- matrix(sims[i,], V, V)
      eigen.chains[i,] <- log10(eigen(covm)$values)
    }
    # percents of explained variability
    exp.vars <- eigen.chains/rowSums(eigen.chains) * 100
    # posteriors of eigenvalues as boxplots
    par(mfrow=c(1,2))
    boxplot(eigen.chains, ylab="Log10 eigenvalue", xlab="PCA axis", 
            col="grey", outline=FALSE)
    # boxplot(exp.vars, ylab="Explained variability [% of total]", xlab="PCA axis", 
    #         col="grey", outline=FALSE, ylim=c(0,100))
    
    results <- list(Eigenvalues = summary(eigen.chains),
                    Exp.var = summary(exp.vars))
    return(results)
} 


sim.bPCA_ard <- function(data, 
                     Q, 
                     mu.prior, 
                     mu.prior.cov, 
                     a_alpha, 
                     b_alpha, 
                     a_tau, 
                     b_tau, 
                     n.chains = 3, 
                     n.iter = 5000, 
                     n.burnin = 4500) {
  # requirements
  require(R2jags)
  require(MASS)
  require(Matrix)
  require(coda)
  
  # dataset dimensions
  N <- nrow(data)
  V <- ncol(data)
  
  # defaults for priors
  if (missing(mu.prior)) mu.prior <- rep(0, V)
  if (missing(mu.prior.cov)) mu.prior.cov <- as.matrix(Diagonal(V, 1000))
  if (missing(a_alpha)) a_alpha <- 1e-3
  if (missing(b_alpha)) b_alpha <- 1e-3
  if (missing(a_tau)) a_tau <- 1e-3
  if (missing(b_tau)) b_tau <- 1e-3
  
  # makes precisions from covariances
  mu.prior.prec <- solve(mu.prior.cov)
  
  # puts data into list
  listdata <- list(Y = as.matrix(data), 
                   N = N,
                   V = V, 
                   Q = Q,
                   mu.prior = mu.prior, 
                   mu.prior.prec = mu.prior.prec,
                   a_alpha = a_alpha,
                   b_alpha = b_alpha,
                   a_tau = a_tau,
                   b_tau = b_tau)
  
  # defines the model in JAGS language
  cat("
  model {
    # Priors for the mean vector
    mu[1:V] ~ dmnorm(mu.prior[], mu.prior.prec[,])
    
    # Priors for the weight matrix W
    for (i in 1:V) {
      for (j in 1:Q) {
        W[i, j] ~ dnorm(0, alpha[j])
      }
    }

    # Priors for precision parameters alpha
    for (j in 1:Q) {
      alpha[j] ~ dgamma(a_alpha, b_alpha)
    }

    # Prior for noise precision tau
    tau ~ dgamma(a_tau, b_tau)
    
    # Priors for latent variables X
    for (i in 1:N) {
      for (j in 1:Q) {
        X[i, j] ~ dnorm(0, 1)
      }
    }

    # Define identity matrix
    for (i in 1:V) {
      for (j in 1:V) {
        I[i, j] <- equals(i, j)
      }
    }
    
    # Likelihood
    for (i in 1:N) {
      mu_pred[i, 1:V] <- mu[] + W[,] %*% X[i,]
      Y[i, 1:V] ~ dmnorm(mu_pred[i,], tau * I[,])
    }
  }
  ", file = "PCA_hierarchical.bugs")
  
  # jags model to estimate covariance matrix distribution
  pcabay <- jags(data = listdata,
                 model.file = "PCA_hierarchical.bugs",
                #  parameters.to.save = c("mu", "W", "alpha", "tau"),
                 parameters.to.save = c("alpha", "tau"),
                 n.chains = n.chains,
                 n.iter = n.iter,
                 n.burnin = n.burnin, 
                 DIC = FALSE)
  
  return(pcabay)
}




simulate2 <- function(psi = 1, N = 100, P = 10) {
  psi_inv <- 1 / psi
  cov <- diag(c(5, 4, 3, 2, rep(psi_inv, P - 4)))
  
  # Generate a random orthogonal matrix U
  random_matrix <- matrix(rnorm(P * P), P, P)
  svd_result <- svd(random_matrix)
  U <- svd_result$u
  
  # Transform the covariance matrix
  cov <- U %*% cov %*% t(U)
  
  y <- mvrnorm(N, mu = rep(0, P), Sigma = cov)
  return(y)
}

set.seed(123)
var_noise <- 0.01
N <- 100
P <- 10
data <- simulate2(psi = 1 / var_noise, N = N, P = P)

covmat_prior <- as.matrix(Diagonal(ncol(data), 1))
covmat_prior_DF <- ncol(data)

result <- sim.bPCA(data = data,
                   covmat.prior = covmat_prior, 
                   covmat.prior.DF = covmat_prior_DF, 
                   n.chains = 3,
                   n.iter = 500,
                   n.burnin = 300,
                   )

eigenval_results <- eigenvalplotslg.bPCA(result, data)
# print(eigenval_results)
# biplots.bPCA(result, data)


start_time <- Sys.time()

# # Run the function with wishart prior
# result <- sim.bPCA(data = data,
#                    covmat.prior = covmat_prior, 
#                    covmat.prior.DF = covmat_prior_DF, 
#                    n.chains = 3,
#                    n.iter = 200,
#                 #    n.iter = 1200,
#                    n.burnin = 100)
# Run the model with hierarchical prior
bpca.fit <- sim.bPCA_ard(data = data, Q = P-1, n.chains = 1, n.iter = 1200, n.burnin = 100)
# plot(-sort(log10(colMeans(bpca_fit$BUGSoutput$sims.matrix[, -ncol(bpca_fit$BUGSoutput$sims.matrix)]))),
#     type = "b", pch = 19, col = "blue",
#     xlab = "Index", ylab = "Log10(Column Means)",
#     main = "Inverse Sorted Log10 of Column Means"
# )

# End timing
end_time <- Sys.time()

# Calculate the elapsed time
elapsed_time <- end_time - start_time

# Print the elapsed time
print(elapsed_time)