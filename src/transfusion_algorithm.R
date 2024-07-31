### (D-)TransFusion Algorithm ###
# Note: this implementation of TransFusion only applies for standardized data with no intercept,
#   raw data should be preprocessed before being fed into the model.
# Note: These functions rely on utility functions defined in `utils.R`. 
# Make sure to source `utils.R` before using these functions.
# Example: source("utils.R")

# A TransFusion framework (He et al. 2024)
TransFusion = function(dataset, vlambda=NULL, debias=F){
  # Input: 
  # dataset: output of a "generate_data_XXX" function. A list consists of target task design matrix X0, 
  #   target task response y0, list of source task design matrix X and list of source task response y.
  #   If one wants to use a self-defined dataset, the data should be PREPROCESSED (standardize & add intercept) before being fed into the model.
  # vlambda: weights of penalty for each feature
  # debias: whether perform the debiased step (the second step)
  
  # Extract target and source data from the input data object
  X0 = dataset$X0
  y0 = dataset$y0
  X = dataset$X
  y = dataset$y
  
  # Stack the source data and the target data following the TF framework
  TF_data = create_TF_data(y, y0, X, X0)
  y_TF = TF_data$y_TF
  X_TF = TF_data$X_TF
  
  # If the penalty vector is not specified, calculate the initial value based on the theoretical results
  if(is.null(vlambda)){
    vlambda = create_vlambda(X, X0)
  }
  
  # Perform LASSO on the combined dataset
  res = lasso_glmnet(X_TF, y_TF, vlambda=vlambda)
  # Store the full vector
  p=ncol(X0); K=length(X);
  beta_full = res$coef
  beta_fullmatr <- matrix(beta_full, nrow = p, ncol = K+1, byrow = FALSE)
  betasourcematr <- cbind(beta_fullmatr[,1:K] + beta_fullmatr[,K+1],beta_fullmatr[,K+1])
  
  # get the weighted average
  beta_full <- append(X, list(X0))
  # Step 1: Calculate the weights vector: from the theory, it's prop to n_k
  weights <- sapply(beta_full, function(mat) nrow(mat))
  # Step 2: Normalize the weights vector
  weights_normalized <- weights / sum(weights)
  # Step 3: Weighted Average
  avesourcebeta <- betasourcematr %*% weights_normalized
  
  # An additional debias step for the target dataset
  if(debias == T){
    # DeBias LASSO
    # Calculate the bias in the target dataset based on the TF estimator
    bias = y0 - X0 %*% avesourcebeta
    # Perform LASSO on the target dataset using the bias
    res_2 = lasso_glmnet(X0, bias)
    # Store the improved estimate
    res$coefplus = avesourcebeta + res_2$coef
  }
  
  # As the result is for the one-step TF vector, we use the average.
  res$coef = avesourcebeta
  
  # Output:
  # res: a list containing the following elements:
  #   coef: final solution vector (beta) after running the one-step algorithm 
  #   coefplus: final solution vector (beta) after running the two-step algorithm
  return(res)
}


# A Distributed TransFusion framework (He et al. 2024)
DTransFusion = function(dataset, estimator_type=c("debiased_lasso", "scad"), vlambda=NULL, debias=F){
  # Input: 
  # dataset: output of a "generate_data_XXX" function. A list consists of target task design matrix X0, 
  #   target task response y0, list of source task design matrix X and list of source task response y.
  #   If one wants to use a self-defined dataset, the data should be PREPROCESSED (standardize & add intercept) before being fed into the model.
  # debias: if TRUE, further perform debiased lasso on the target dataset  

  estimator_type = match.arg(estimator_type)  

  # Extract target and source data from the input data object
  X0 = dataset$X0
  y0 = dataset$y0
  X = dataset$X
  y = dataset$y
  
  # Perform debiased lasso
  y_d = list()
  x_d = list()
  
  K = length(X)
  for(k in 1:K){
    Xk = X[[k]]
    yk = y[[k]]
    nk = nrow(Xk)
    pk = ncol(Xk)
    sigma.hat = (1/nk)*(t(Xk)%*%Xk)
    # lasso
    res_k = lasso_glmnet(Xk, yk)
    if(estimator_type=="debiased_lasso"){
        U <- InverseLinfty(sigma.hat, nk)
        beta_debiased = as.numeric(res_k$coef + (U%*%t(Xk)%*%(yk - Xk %*% res_k$coef))/nk)
        # create the new X and y based on the D-TransFusion formulation
        y_d[[k]] = sqrt(nk) * beta_debiased
        x_d[[k]] = sqrt(nk) * diag(pk)
    }else{
        # For an efficient implementation of SCAD, we use a simplified LLA (local linear approximation) with a = 3.7.
        res_k = lasso_glmnet(Xk, yk) # Fit a lasso regression to obtain the initial choice of lambda
        init_lambda = res_k$best_lambda
        vlambda_d = rep(init_lambda, ncol(Xk)) # Calculate the approximate SCAD penalty function using the idea of LLA
        vlambda_d[abs(res_k$coef) > 3.7 * init_lambda] = 0
        res_k_scad = lasso_glmnet(Xk, yk, vlambda=vlambda_d) # Apply the new penalty function
        y_d[[k]] = sqrt(nk) * as.numeric(res_k_scad$coef)
        x_d[[k]] = sqrt(nk) * diag(pk)
    }
  }
  
  # Stack the source data (summary statistics) and the target data following the TF framework
  TF_data = create_TF_data(y_d, y0, x_d, X0)
  y_TF = TF_data$y_TF
  X_TF = TF_data$X_TF
  
  # If the penalty vector is not specified, calculate the initial value based on the theoretical results
  if(is.null(vlambda)){
    vlambda = create_vlambda(X, X0)
  }
  
  # Perform LASSO on the combined dataset
  res = lasso_glmnet(X_TF, y_TF, vlambda=vlambda)
  # Store the full vector
  p=ncol(X0); K=length(X);
  beta_full = res$coef
  beta_fullmatr <- matrix(beta_full, nrow = p, ncol = K+1, byrow = FALSE)
  betasourcematr <- cbind(beta_fullmatr[,1:K] + beta_fullmatr[,K+1],beta_fullmatr[,K+1])
  
  # get the weighted average
  beta_full <- append(X, list(X0))
  # Step 1: Calculate the weights vector: from the theory, it's prop to n_k
  weights <- sapply(beta_full, function(mat) nrow(mat))
  # Step 2: Normalize the weights vector
  weights_normalized <- weights / sum(weights)
  # Step 3: Weighted Average
  avesourcebeta <- betasourcematr %*% weights_normalized
  
  # An additional debias step for the target dataset
  if(debias == T){
    # DeBias LASSO
    # Calculate the bias in the target dataset based on the TF estimator
    bias = y0 - X0 %*% avesourcebeta
    # Perform LASSO on the target dataset using the bias
    res_2 = lasso_glmnet(X0, bias)
    # Store the improved estimate
    res$coefplus = avesourcebeta + res_2$coef
  }
  
  # As the result is for the one-step TF vector, we use the average.
  res$coef = avesourcebeta
  
  # Output:
  # res: a list containing the following elements:
  #   coef: final solution vector (beta) after running the one-step algorithm 
  #   coefplus: final solution vector (beta) after running the two-step algorithm
  return(res)
}
