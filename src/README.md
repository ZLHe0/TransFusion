# TransFusion: Covariate-Shift Robust Transfer Learning for High-Dimensional Regression

This repository contains the code to implement the proposed method in "TransFusion: Covariate-Shift Robust Transfer Learning for High-Dimensional Regression" published at AISTAT 2024.

## Description

TransFusion is designed to tackle model shifts in the presence of covariate shifts in the high-dimensional regression setting. The method is also extended to a distributed setting, requiring just one round of communication while retaining the estimation rate of the centralized version. Numerical results validate our theory and highlights the method's robustness to covariate shifts.

At the technical level, the proof of the theorem involves a non-asymptotic analysis of a coupled structure in the fused regularizer, together with a fine treatment of both the non-strong convexity and
smoothness, which may be of independent interest.

## Getting Started

### Dependencies

Before running the code, ensure you have the following prerequisites installed:

- R
- Libraries:
  - glmnet
  - ggplot2
  - MASS
  - parallel

You can install the required libraries in R using the following commands:

```r
install.packages("glmnet")
install.packages("ggplot2")
install.packages("MASS")
install.packages("parallel")
```

### Structure

The project is organized as follows:
```
project_root/
├── src/
│   ├── transfusion_algorithm.R   # Implementation of (D-)TransFusion
│   ├── baseline_algorithm.R      # Implementation of Baseline Methods
│   ├── data_generation.R         # Functions for generating simulation data
│   └── utils.R                   # Utility functions
├── run_transfusion_exp.R          # Script to generate results shown in Figure 1
├── run_dtransfusion_exp.R         # Script to generate results shown in Figure 2
└── README.md   
```                       

### Executing Program

To run the experiments for the results shown in Figure 1 and Figure 2 of the paper, use the following scripts:

#### Running TransFusion Experiments

To run the TransFusion experiments:

```r
source("run_transfusion_exp.R")
```

#### Running D-TransFusion Experiments

To run the D-TransFusion experiments:

```r
source("run_dtransfusion_exp.R")
```

## Citation
We kindly ask anyone uses the method/code/proof in your research to cite the paper. We welcome any discussions, suggestions, and pointing out of issues.

```
@inproceedings{he2024transfusion,
  title={Transfusion: Covariate-shift robust transfer learning for high-dimensional regression},
  author={He, Zelin and Sun, Ying and Li, Runze},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={703--711},
  year={2024},
  organization={PMLR}
}
```

Additionally, the method AdaTrans from our paper *"AdaTrans: Feature-wise and Sample-wise Adaptive Transfer Learning for High-dimensional Regression"* can be similarly implemented, and the code will be updated later.


## Contact

Contributor: Zelin He
 [@zbh5185](zbh5185@psu.edu)


## License

MIT