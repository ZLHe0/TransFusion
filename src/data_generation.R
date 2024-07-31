# Data Generation Tools
# Note: These functions rely on utility functions defined in `utils.R`. 
# Make sure to source `utils.R` before using these functions.
# Example: source("utils.R")

generate_data = function(hk_strength, K, spar = 100, p = 500, n0=150, n=NULL, s=16,
                                  cov_func=c('toeplitz_covariance', 'weighted_covariance', 'random_covariance', 'identity_covariance'),
                                  abs_signal = T, exactly_diverse = F,
                                  random_assign = F) {
  # Input:
  # hk_strength: strength of non-transferable signal (hk)
  # spar: sparsity of the non-transferable signal
  # K: the number of source datasets
  # n0: the number of samples in the target dataset (default: 150)
  # n: a vector of sample sizes for each source dataset (default: rep(100, K))
  # p: the number of features (default: 500)
  # s: the number of non-zero elements in the true coefficient vector (default: 16)
  # cov_func: the function to generate covariance matrix for the source data, can be one of:
  #           'toeplitz_covariance', 'weighted_covariance', 'random_covariance', 'identity_covariance'
  # abs_signal: boolean indicating whether to add a non-zero mean to the noise
  # exactly_diverse: boolean indicating whether to enforce exact diversity in source tasks
  # random_assign: boolean indicating whether to randomly assign noise to features

  if (is.null(n)) {
    n = rep(100, K)
  }
  
  # Generate beta
  beta = c(rep(0.3, s), rep(0, p - s))
  
  # Generate target data
  X0 = matrix(rnorm(n0 * p), nrow = n0, ncol = p)
  eps0 = rnorm(n0)
  y0 = X0 %*% beta + eps0
  
  # Generate source data
  X = list()
  y = list()
  w = list()
  wsum_temp = rep(0,p) # For diverse source task case
  
  for (k in 1:K) {
    # set.seed(k) # Make sure source sets are different
    
    # Heterogeneous X
    Sigma = cov_func(p, k, K)
    X_k = MASS::mvrnorm(n = n[k], mu = rep(0, p), Sigma = Sigma)
    X[[k]] = X_k
    eps = rnorm(n[k])
    
    # If take a non-zero mean to the noises
    if(abs_signal == T){
      xi = rnorm(p,mean=0.1, sd=hk_strength/spar)
    }else{
      xi = rnorm(p,mean=0, sd=hk_strength/spar)
    }
    
    # If random assign noises
    if(random_assign == T){
      H_k = sample(1:p, spar, replace = FALSE)
      w_k = beta + xi * (1:p %in% H_k)
    }else{
      H_k = 1:spar
      w_k = beta + xi * (1:p %in% H_k)
    }
    
    w_k[1] = -0.3
    w[[k]] = w_k
    
    if(exactly_diverse == T){
      wsum_temp = wsum_temp + w_k
      if(k == K && k > 1){
        w[[k]] = K * beta - wsum_temp
      }
    }
    
    y[[k]] = X[[k]] %*% w[[k]] + eps
  }
  
  # Output:
  # X0: target design matrix
  # y0: target response vector
  # X: list of source design matrices
  # y: list of source response vectors
  # w: list of source coefficient vectors
  # beta: true coefficient vector for the target data
  return(list(X0 = X0, y0 = y0, X = X, y = y, w = w, beta = beta))
}


### Covariance matrices generation
# Function to calculate toeplitz matrix
toeplitz_covariance = function(p, k, K) {
  first_row = sapply(0:(p-1), function(i) (k/(K+2)) ^ i)
  Sigma = toeplitz(first_row)
  return(Sigma)
}

# Function to calculate weighted identity matrix
weighted_covariance = function(p, k, K) {
  Sigma = k*diag(p)
  return(Sigma)
}

# Function to generate random matrix
random_covariance = function(p, k, K) {
  # Generate A^(k) with given probabilities
  A_k = matrix(rbinom(p*p, 1, 0.3) * 0.3, ncol=p)
  
  # Calculate Sigma^(k)
  sigma_k = t(A_k) %*% A_k + diag(p)
  
  return(sigma_k)
}

# Function to calculate identity covariance matrix
identity_covariance = function(p, k, K) {
  # Create an identity matrix of size p x p
  Sigma = diag(p)
  return(Sigma)
}