##### main file to run transfusion-related experiments (results in Figure 1)
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

# Set the range of K values
K_values = c(1,3,5,7,9,11,13,15,17,19)

# hk_strength: strength of non-transferable signal (hk)
hk_strength = 4
# spar: sparsity of the non-transferable signal
spar =  50

# Set the number of replicates
num_replicates = 100

# Set parameter
n0 = 150 # Target sample size
p = 500 # Feature Dimension
s = 16 # Sparisity Level

# Initialize empty lists to store l-2 norm errors
trans_errors = list()
agg_errors = list()
TransHDGLM_errors = list()
tf_trans_errors = list()
tf2_trans_errors = list()
single_errors = list()

# Generate data and perform Trans_lasso, Agg_lasso, Trans_HDGLM, ULM(_Plus)_Trans_lasso for each K value
for (K in K_values) {
  # Equal sample size
  n = rep(200, K)
  
  # Initialize errors for the current K value
  trans_error_K = 0
  agg_error_K = 0
  TransHDGLM_error_K = 0
  tf_trans_error_K = 0
  tf2_trans_error_K = 0
  single_error_K = 0  
  
  for (replicate in 1:num_replicates) {
    # Generate data
    data = generate_data(hk_strength = hk_strength, K = K,
                                spar = spar, n0=n0, n=n, p=p, s=s,
                                abs_signal=T, exactly_diverse=F, cov_func=random_covariance)
    
    # Perform Trans_lasso
    trans_result = Trans_lasso(dataset = data)
    trans_error_K = trans_error_K + l2_norm_error(trans_result$coef, data$beta)
    
    # Perform Agg_lasso
    agg_result = Agg_lasso(dataset = data)
    agg_error_K = agg_error_K + l2_norm_error(agg_result$coef, data$beta)
    
    # Perform Single_lasso
    single_result = Single_lasso(dataset = data)
    single_error_K = single_error_K + l2_norm_error(single_result$coef, data$beta)
    
    # Perform Trans_HDGLM
    TransHDGLM_result = TransHDGLM(dataset = data)
    TransHDGLM_error_K = TransHDGLM_error_K + l2_norm_error(TransHDGLM_result$coef, data$beta)
    
    # Perform TransFusion_onestep
    tf_trans_result = TransFusion(dataset = data, debias=T)
    tf_trans_error_K = tf_trans_error_K + l2_norm_error(tf_trans_result$coef, data$beta)
    
    # Perform TransFusion_twostep
    tf2_trans_error_K = tf2_trans_error_K + l2_norm_error(tf_trans_result$coefplus, data$beta)
    
  }
  
  # Average the errors over the replicates
  single_errors = append(single_errors, single_error_K / num_replicates)
  trans_errors = append(trans_errors, trans_error_K / num_replicates)
  agg_errors = append(agg_errors, agg_error_K / num_replicates)
  TransHDGLM_errors = append(TransHDGLM_errors, TransHDGLM_error_K / num_replicates)
  tf_trans_errors = append(tf_trans_errors, tf_trans_error_K / num_replicates)
  tf2_trans_errors = append(tf2_trans_errors, tf2_trans_error_K / num_replicates)
}

# Update the data frame for plotting
plot_data = data.frame(
  K_val = rep(K_values, 6), 
  Error = c(unlist(trans_errors),
            unlist(agg_errors),
            unlist(TransHDGLM_errors),
            unlist(tf_trans_errors),
            unlist(tf2_trans_errors),
            unlist(single_errors)),  # Include single_errors here
  Method = factor(c(rep("Trans_lasso", length(K_values)),
                    rep("Agg_lasso", length(K_values)),
                    rep("Trans_HDGLM", length(K_values)),
                    rep("TransFusion_onestep", length(K_values)),
                    rep("TransFusion_twostep", length(K_values)),
                    rep("Single_lasso", length(K_values))),
                  # Include Single_lasso here
                  levels = c("Trans_lasso",
                             "Agg_lasso",
                             "Trans_HDGLM",
                             "TransFusion_onestep", 
                             "TransFusion_twostep",
                             "Single_lasso"))  # Include Single_lasso in levels
)

# Plot the l-2 norm errors
(l2errorplot = ggplot(plot_data, aes(x = K_val, y = Error, color = Method)) +
    geom_line() +
    geom_point() +
    labs(title = paste0("L-2 Norm Error Comparison"),
         x = "Number of Source Tasks (K)",
         y = "L-2 Norm Error"))

### Save the results

# Define the directories to save the files
figures_directory = "Simulation_Figures"
data_directory = "Simulation_Data"

# Create the directories if they do not exist
dir.create(figures_directory, showWarnings = FALSE)
dir.create(data_directory, showWarnings = FALSE)

# Save the results
# Save the figure
ggsave(filename = paste0(figures_directory, "/transfusion_result_with_hk", hk_strength, "and_sparsity", spar, ".png"), plot = l2errorplot)
# Save the plot_data dataframe as a CSV file
write.csv(plot_data, file = paste0(data_directory, "/transfusion_result_with_hk", hk_strength, "and_sparsity", spar, ".csv"), row.names = FALSE)


