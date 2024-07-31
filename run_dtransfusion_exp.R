##### main file to run D-transfusion-related experiments (results in Figure 2)
# Note: this implementation of TransFusion only applies for standardized data with no intercept,
#   raw data should be preprocessed before being fed into the model.

rm(list=ls())

# Implement LASSO
library(glmnet)
# Packages for plotting figures
library(ggplot2)
# Math Calculation Package
library(MASS)
# Perallel Computing
library(parallel)
# Utility functions
source('src/utils.R')
# Data generation functions
source('src/data_generation.R')
# Algorithm implementations
source('src/baseline_algorithm.R')
source('src/transfusion_algorithm.R')

# Define a function to calculate the l-2 norm^2 error
l2_norm_error = function(x, beta) {
  return(sum((x - beta)^2))
}

# Set the range of K values
K_values = c(1,3,5,7,10)

# hk_strength: strength of non-transferable signal (hk)
hk_strength = 4
# spar: sparsity of the non-transferable signal
spar =  50

# Set the number of replicates
num_replicates = 100

# Set parameter
n0 = 150 # Target sample size
p = 500 # Dimension
s = 16 # Sparisity Level

# Initialize empty lists to store l-2 norm errors
ctf_trans_errors = list()
ctf_plus_trans_errors = list()
tf_trans_errors = list()
tf2_trans_errors = list()
single_errors = list()

# Generate data and perform methods for each K value
for (K in K_values) {
  n = rep(200, K)
  
  # Initialize errors for the current K value
  ctf_trans_error_K = 0
  ctf_plus_trans_error_K = 0
  tf_trans_error_K = 0
  tf2_trans_error_K = 0
  single_error_K = 0  
  
  for (replicate in 1:num_replicates) {
    # Generate data
    data = generate_data(hk_strength = hk_strength, K = K,
                                  spar = spar, n0=n0, n=n, p=p, s=s,
                                  abs_signal=T, exactly_diverse=F, cov_func=random_covariance)
    
    # Perform DTransFusion onestep
    # replacing the estimator with `scad` estimator leads to a similar performance but much faster computation
    ctf_trans_result = DTransFusion(dataset = data, estimator_type="debiased_lasso", debias=T) 
    ctf_trans_error_K = ctf_trans_error_K + l2_norm_error(ctf_trans_result$coef, data$beta)
    
    # Perform DTransFusion twostep
    ctf_plus_trans_error_K = ctf_plus_trans_error_K + l2_norm_error(ctf_trans_result$coefplus, data$beta)  
    
    # Perform Single_lasso
    single_result = Single_lasso(dataset = data)
    single_error_K = single_error_K + l2_norm_error(single_result$coef, data$beta)
    
    # Perform TransFusion onestep
    tf_trans_result = TransFusion(dataset = data, debias=T)
    tf_trans_error_K = tf_trans_error_K + l2_norm_error(tf_trans_result$coef, data$beta)
    
    # Perform TransFusion twostep
    tf2_trans_error_K = tf2_trans_error_K + l2_norm_error(tf_trans_result$coefplus, data$beta)
    
  }
  
  single_errors = append(single_errors, single_error_K / num_replicates)
  ctf_trans_errors = append(ctf_trans_errors, ctf_trans_error_K / num_replicates)
  ctf_plus_trans_errors = append(ctf_plus_trans_errors, ctf_plus_trans_error_K / num_replicates)
  tf_trans_errors = append(tf_trans_errors, tf_trans_error_K / num_replicates)
  tf2_trans_errors = append(tf2_trans_errors, tf2_trans_error_K / num_replicates)
}

# Update the data frame for plotting
plot_data = data.frame(
  K_val = rep(K_values, 5),
  Error = c(unlist(ctf_trans_errors), 
            unlist(ctf_plus_trans_errors),
            unlist(tf_trans_errors),
            unlist(tf2_trans_errors),
            unlist(single_errors)),  # Include single_errors here
  Method = factor(c(rep("DTransFusion_onestep", length(K_values)),
                    rep("DTransFusion_twostep", length(K_values)),
                    rep("TransFusion_onestep", length(K_values)),
                    rep("TransFusion_twostep", length(K_values)),
                    rep("Single_lasso", length(K_values))), 
                  levels = c("DTransFusion_onestep",
                             "DTransFusion_twostep",
                             "TransFusion_onestep", 
                             "TransFusion_twostep",
                             "Single_lasso"))
)

# Plot the l-2 norm errors
l2errorplot = ggplot(plot_data, aes(x = K_val, y = Error, color = Method)) +
  geom_line() +
  geom_point() +
  labs(title = paste0("L-2 Norm Error Comparison"),
       x = "Number of Source Tasks (K)",
       y = "L-2 Norm Error")

### Save the results

# Define the directories to save the files
figures_directory = "Simulation_Figures"
data_directory = "Simulation_Data"

# Create the directories if they do not exist
dir.create(figures_directory, showWarnings = FALSE)
dir.create(data_directory, showWarnings = FALSE)

# Save the results
ggsave(filename = paste0(figures_directory, "Dtransfusion_result_with_hk", hk_strength, "and_sparsity", spar, ".png"), plot = l2errorplot)
write.csv(plot_data, file = paste0(data_directory, "Dtransfusion_result_with_hk", hk_strength, "and_sparsity", spar, ".csv"), row.names = FALSE)