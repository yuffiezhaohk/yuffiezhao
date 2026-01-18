# **DeepHalo and Sparse Shocks: Replication of Discrete Choice Models**

## **🎯 Project Goal**

This project consists of two primary research components:
1.  **DeepHalo Replication:** A replication of the **featureless** DeepHalo model introduced by Zhang et al. (2025), focusing on neural network-based Discrete Choice Models (DCM) and RMSE performance.
2.  **Sparse Shocks Lu and Shimizu (2025):** An replication of Lu and Shimizu (2025), implementing a different econometric approach to choice modeling, specifically focusing on sparse shocks and shrinkage estimators.

## **📂 File Structure and Descriptions**

This repository contains the necessary files for synthetic data generation, model training, quality assurance, and project documentation for both research phases.

### **Phase 1: DeepHalo (Based on Zhang et al., 2025)**
| Filename | Type | Description |
| :---- | :---- | :---- |
| synthetic\_data\_deephalo.ipynb | Jupyter Notebook | **Full Experiment Script.** Complete code for data generation, model construction, and the full 500-epoch training run. |
| synthetic\_data\_deephalo\_epoch100.ipynb | Jupyter Notebook | **Rapid Verification Script.** A version optimized for 100 epochs to perform stability checks. |
| testing\_plan.md | Markdown | **QA Strategy.** Defines the scope and mathematical constraints for the DeepHalo architecture. |
| testing\_report.md | Markdown | **Test Execution.** Outcomes of Unit and Integration Tests on Google Colab (T4 GPU). |
| report.pdf | PDF | **Research Report.** Comprehensive document covering DeepHalo methodology and literature review. |

### **Phase 2: Sparse Shocks & Shrinkage (Based on Lu and Shimizu 2025)**
| Filename | Type | Description |
| :---- | :---- | :---- |
| **part2-sparseshocks.ipynb** | Jupyter Notebook | **Core Research Script.** Implements the Lu and Shimizu (2025) methodology, including BLP-style simulation, GMM estimation, and Bayesian MCMC shrinkage estimators. |
| **part2-testing plan and report.md** | Markdown | **QA Documentation.** Combined testing plan and execution report verifying the mathematical integrity of the BLP pipeline and MCMC samplers. |
| **part2-report.pdf** | PDF | **Technical Report.** Detailed discussion on the implementation of Lu (2025), sparse shock structures, and experimental findings. |


## **✅ Testing and Validation Status**

### **DeepHalo (Phase 1)**
* **Mathematical Constraints:** `calculate_layer_width` logic is confirmed sound.
* **Architecture:** **Quadratic Activation** and **Availability Masking** layers are correctly implemented.

### **Sparse Shocks (Phase 2)**
* **DGP Integrity:** Simulation correctly generates $(T \times N, J+1)$ choice sets.
* **Estimator Stability:** BLP GMM objective function and Bayesian MCMC chains are validated for convergence and dimensionality.

## **⚙️ Running the Project**

1.  **Environment Setup:** Google Colab is recommended. Required libraries: `TensorFlow/Keras`, `NumPy`, `SciPy`, and `Matplotlib`.
2.  **Phase 1 Execution:** Run `synthetic_data_deephalo.ipynb` for the neural network-based choice model results.
3.  **Phase 2 Execution:** Run `part2-sparseshocks.ipynb` to execute the BLP estimation and Bayesian shrinkage experiments.