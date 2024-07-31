### Baseline algorithms ###
# Note: These functions rely on utility functions defined in `utils.R`. 
# Make sure to source `utils.R` before using these functions.
# Example: source("utils.R")

### Single-Lasso
# Perform a Lasso regression on a single task
Single_lasso = function(dataset){
  # Input: 
  # dataset: output of a "generate_data_XXX" function. A list consists of target task design matrix X0, target task response y0, list of source task design matrix X and list of source task response y

  # Extract target and source data from the input data object
  X0 = dataset$X0
  y0 = dataset$y0
  
  res = lasso_glmnet(X0, y0)
  
  # Output:
  # res: a list containing the following elements:
  #  beta: final solution vector (beta) after running the algorithm
  
  return(res)
}

### Agg-Lasso
# Aggregate all the data together then perform a Lasso regression on the aggregated dataset
Agg_lasso = function(dataset){
  # Input: 
  # dataset: output of a "generate_data_XXX" function. A list consists of target task design matrix X0, target task response y0, list of source task design matrix X and list of source task response y
  
  # Extract target and source data from the input data object
  X0 = dataset$X0
  y0 = dataset$y0
  X = dataset$X
  y = dataset$y
  
  # Combine target and source data more efficiently
  X_combined <- do.call(rbind, c(list(X0), X))
  y_combined <- do.call(c, c(list(y0), y))
  
  res = lasso_glmnet(X_combined, y_combined)
  
  # Output:
  # res: a list containing the following elements:
  #  beta: final solution vector (beta) after running the algorithm
  
  return(res)
}


### (Naive) Trans-Lasso (Li et al. 2022)
# A two-step approach to achieve transfer learning
Trans_lasso = function(dataset){
  # Input: 
  # dataset: output of a "generate_data_XXX" function. A list consists of target task design matrix X0, target task response y0, list of source task design matrix X and list of source task response y
  
  # Extract target and source data from the input data object
  X0 = dataset$X0
  y0 = dataset$y0
  X = dataset$X
  y = dataset$y
  
  # Step 1: Aggregated LASSO
  # Combine target and source data more efficiently
  X_combined <- do.call(rbind, c(list(X0), X))
  y_combined <- do.call(c, c(list(y0), y))
  # Perform LASSO on the combined dataset
  res_1 = lasso_glmnet(X_combined, y_combined)
  
  # Step 2: DeBias LASSO
  # Calculate the bias in the target dataset based on the pooled estimator from the first step
  bias = y0 - X0 %*% res_1$coef
  # Perform LASSO on the target dataset using the bias
  res_2 = lasso_glmnet(X0, bias)
  
  # Step 3: Combine results, get the refined estimator
  res = list()
  res$coef = res_1$coef + res_2$coef

  
  # Output:
  # res: a list containing the following elements:
  #  beta: final solution vector (beta) after running the algorithm
  return(res)
}


### (Naive) TransHDGLM (Li et al. 2023)
# An alternating two-step algorithm to achieve transfer learning
TransHDGLM = function(dataset, eps = 10e-2, max_iter=10){
  # Input: 
  # dataset: output of a "generate_data_XXX" function. A list consists of target task design matrix X0, target task response y0, list of source task design matrix X and list of source task response y
  # eps: maximum value of optimization uncertainty for stopping
  # max_iter: maximum number of iterations
  
  # Extract target and source data from the input data object
  X0 = dataset$X0
  y0 = dataset$y0
  X = dataset$X
  y = dataset$y
  
  # Iteratively update delta and beta
  # Initialization
  X_combined = do.call(rbind, c(list(X0), X))
  y_combined = do.call(c, c(list(y0), y))
  K = length(X)
  temp = rep(0, ncol(X0))
  # Perform LASSO on the combined dataset
  res_1 = lasso_glmnet(X_combined, y_combined)
  
  for(i in 1:max_iter){
    # Step 1: DeBias LASSO
    # Calculate the bias in the source datasets based on the pooled estimator from the first step
    res_2 = list() # Store the result for each source dataset in step 2
    # Define a function to be executed
    perform_lasso_on_source = function(k) {
      bias_k = y[[k]] - X[[k]] %*% res_1$coef
      # Perform LASSO on the source dataset using the bias
      res_2_k = lasso_glmnet(X[[k]], bias_k)
      return(res_2_k)
    }
    # Execute the function in parallel for each value of k
    res_2 = lapply(1:K, perform_lasso_on_source)
    
    # Step 2: Combine results, run the de-biased pooled LASSO
    for(k in 1:K){
      y[[k]] = y[[k]] - X[[k]] %*% res_2[[k]]$coef
    }
    # Combine target and source data more efficiently
    X_combined = do.call(rbind, c(list(X0), X))
    y_combined_new = do.call(c, c(list(y0), y))
    # Perform LASSO on the combined dataset
    res_3 = lasso_glmnet(X_combined, y_combined_new)
    
    if(sum((res_3$coef - temp)^2) < eps){
      break
    }else{
      # Store beta
      temp = res_3$coef
      # Update beta
      res_1 = res_3
    }
  }
  
  res = list()
  res$coef = res_3$coef
  
  # Output:
  # res: a list containing the following elements:
  #  beta: final solution vector (beta) after running the algorithm
  return(res)
}