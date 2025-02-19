### Utils ###
library(glmnet)

## 1 Optimization Tools

lasso_glmnet = function(X, y, vlambda=NULL) {
  # This function performs Lasso regression using the glmnet package.
  # Inputs:
  # - X: A matrix of predictor variables.
  # - y: A vector of response variables.
  # - vlambda: An optional vector containing the weight for the penalty for each feature in X.
  #            If not provided, features are given equal weight.
  #
  # Output:
  # - A list containing the following element:
  #   - coef: The estimated coefficients from the Lasso regression model.
  # Remark: no intercept and standardization is implemented.
  
  # If vlambda is not given, then features are given equal weight
  if (is.null(vlambda)) {
    nvars = ncol(X)
    penalty_factor = rep(1, nvars)
  }else{
    penalty_factor = vlambda
  }
  
  # Normalizing the weight
  penalty_factor = penalty_factor / max(penalty_factor)
  
  # Fit a lasso model using glmnet
  cv_fit = cv.glmnet(X, y, alpha=1, intercept=FALSE, standardize=FALSE,
                      penalty.factor=penalty_factor, nfolds = 10,
                      lambda.min.ratio=0.01)
  # Obtain the best choice of lambda
  best_lambda_min = cv_fit$lambda.min
  
  # Extract the coefficients
  beta = coef(cv_fit, s = "lambda.min")[-1]
  
  # Return the results
  return(list(coef = beta, best_lambda = best_lambda_min))
}

lasso_pgd = function(X, y, vlambda=NULL) {
  # This function performs Lasso regression using the Proximal Gradient Descent (PGD) algorithm.
  # 
  # Note: This method is not as well optimized compared to the glmnet package.
  return(list(coef = NULL))
}



## 2 TransFusion Algorithm Utilities

# Using change of variable to transform the TransFusion problem into a standard weighted Lasso probelm
create_TF_data = function(y, y0, X, X0){
  # Input:
  # X: a list of source design matrices
  # X0: the target design matrix
  # y: a list of source design responses
  # y0: the target response
  # Output:
  # y_TF: Transformed y following the TF framework
  # X_TF: Transformed X following the TF framework
  
  ### Stack X 
  K = length(X)
  
  # Add X0 to the end of the list
  Xapp = X
  Xapp[[K+1]] = X0
  
  # Get the dimensions of each matrix in the list
  dimensions = lapply(Xapp, dim)
  
  # Create a list of matrices filled with zeros
  zeros_list = lapply(dimensions, function(d) matrix(0, nrow = d[1], ncol = d[2]))
  
  # Initialize an empty list to store the final matrix
  X_final = list()
  
  # Loop through each element in the list, adding the appropriate matrices
  for (i in 1:(K + 1)) {
    row_list = list()
    for (j in 1:(K + 1)) {
      if (j == i) {
        row_list[[j]] = Xapp[[i]]
      } else if(j == K + 1) {
        row_list[[j]] = Xapp[[i]]
      } else {
        row_list[[j]] = zeros_list[[i]]
      }
    }
    X_final[[i]] = do.call(cbind, row_list)
  }
  
  # Combine the rows to get the final matrix
  X_final = do.call(rbind, X_final)
  
  ### Stack y
  y[[K+1]] = y0
  y_final = do.call(c, y)
  
  res = list()
  res$y_TF = y_final
  res$X_TF = X_final
  return(res)
}

### Function to create the default penalty vector for the TF data
create_vlambda = function(X, X0) {
  # Input
  # X: a list contains the source data design matrix. It can be generated by the function generate_data_XXX.
  # X0: target data design matrix. It can be generated by the function generate_data_XXX.
  # Output:
  # result: the penalty vector based on theoretical results.
  
  N = sum(sapply(X, nrow)) + nrow(X0)
  
  # Calculate the theoretical penalty values n_k/N * (\sqrt{\log p / n_k}) for each block
  values = sapply(X, function(x) nrow(x)/N * sqrt(log(ncol(x)) / nrow(x)))
  
  # Create the result vector by repeating each value n_k times
  result = unlist(lapply(1:length(X), function(i) rep(values[i], ncol(X[[i]]))))
  
  # For the target task estimation problem, set lambda to be the global one (\sqrt{\log p/N})
  global_lambda = sqrt( log(ncol(X0)) / N )
  result = c(result, rep(global_lambda, ncol(X0)))

  return(result)
}

### Functions for de-biased Lasso (Javanmard and Montanari, 2014)
# Code to solve the debiased lasso optimization problem
InverseLinftyOneRow = function ( sigma, i, mu, maxiter=50, threshold=1e-2 ) {
  p = nrow(sigma);
  rho = max(abs(sigma[i,-i])) / sigma[i,i];
  mu0 = rho/(1+rho);
  beta = rep(0,p);
  
  if (mu >= mu0){
    beta[i] = (1-mu0)/sigma[i,i];
    returnlist = list("optsol" = beta, "iter" = 0);
    return(returnlist);
  }
  
  diff.norm2 = 1;
  last.norm2 = 1;
  iter = 1;
  iter.old = 1;
  beta[i] = (1-mu0)/sigma[i,i];
  beta.old = beta;
  sigma.tilde = sigma;
  diag(sigma.tilde) = 0;
  vs = -sigma.tilde%*%beta;
  
  while ((iter <= maxiter) && (diff.norm2 >= threshold*last.norm2)){    
    
    for (j in 1:p){
      oldval = beta[j];
      v = vs[j];
      if (j==i)
        v = v+1;    
      beta[j] = SoftThreshold(v,mu)/sigma[j,j];
      if (oldval != beta[j]){
        vs = vs + (oldval-beta[j])*sigma.tilde[,j];
      }
    }
    
    iter = iter + 1;
    if (iter==2*iter.old){
      d = beta - beta.old;
      diff.norm2 = sqrt(sum(d*d));
      last.norm2 =sqrt(sum(beta*beta));
      iter.old = iter;
      beta.old = beta;
      if (iter>10)
        vs = -sigma.tilde%*%beta;
    }
  }
  
  returnlist = list("optsol" = beta, "iter" = iter)
  return(returnlist)
}

# Precision Matrix Estimator
InverseLinfty = function(sigma, n, resol=1.5, mu=NULL, maxiter=50, threshold=1e-2, verbose = TRUE) {
  isgiven = 1;
  if (is.null(mu)){
    isgiven = 0;
  }
  
  p = nrow(sigma);
  M = matrix(0, p, p);
  xperc = 0;
  xp = round(p/10);
  for (i in 1:p) {
    if ((i %% xp)==0){
      xperc = xperc+10;
      if (verbose) {
        print(paste(xperc,"% done",sep="")); }
    }
    if (isgiven==0){
      mu = (1/sqrt(n)) * qnorm(1-(0.1/(p^2)));
    }
    mu.stop = 0;
    try.no = 1;
    incr = 0;
    while ((mu.stop != 1)&&(try.no<10)){
      last.beta = beta
      output = InverseLinftyOneRow(sigma, i, mu, maxiter=maxiter, threshold=threshold)
      beta = output$optsol
      iter = output$iter
      if (isgiven==1){
        mu.stop = 1
      }
      else{
        if (try.no==1){
          if (iter == (maxiter+1)){
            incr = 1;
            mu = mu*resol;
          } else {
            incr = 0;
            mu = mu/resol;
          }
        }
        if (try.no > 1){
          if ((incr == 1)&&(iter == (maxiter+1))){
            mu = mu*resol;
          }
          if ((incr == 1)&&(iter < (maxiter+1))){
            mu.stop = 1;
          }
          if ((incr == 0)&&(iter < (maxiter+1))){
            mu = mu/resol;
          }
          if ((incr == 0)&&(iter == (maxiter+1))){
            mu = mu*resol;
            beta = last.beta;
            mu.stop = 1;
          }                        
        }
      }
      try.no = try.no+1
    }
    M[i,] = beta;
  }
  return(M)
}

# ST tool
SoftThreshold = function( x, lambda ) {
  #
  # Standard soft thresholding
  #
  if (x>lambda){
    return (x-lambda);}
  else {
    if (x< (-lambda)){
      return (x+lambda);}
    else {
      return (0); }
  }
}

# Define a function to calculate the l-2 norm^2 error
l2_norm_error = function(x, beta) {
  return(sum((x - beta)^2))
}



