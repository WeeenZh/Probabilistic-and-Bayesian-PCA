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


simulate2 <- function(psi = 1) {
  psi_inv <- 1 / psi
  N <- 100
  P <- 10
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
var_noise = 0.1
data <- simulate2(psi = 1 / var_noise)

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
