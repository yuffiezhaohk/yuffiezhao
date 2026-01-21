# Sparse Shocks Model Replication

This repository contains a replication of the model introduced by Lu and Shimizu (2025). The system is built using **TensorFlow** and **Choice-Learn**.  

## 📂 Repository Structure  

### 1. Root Directory  
Contains documentation and reports:  

### Files & Folders  
| Name                      | Type       | Description                                  |  
|---------------------------|------------|----------------------------------------------|  
| `code/`                   | Folder     | Core implementation scripts (see below)      |  
| `report.pdf`              | File       | Formal project summary & BLP estimation analysis |  
| `testing plan & report.md`| File       | Validation strategy & execution log |  

### 2. Code Directory (`/code`)  
Contains core implementation files:  

### Files List  
| Name                   | Type       | Description                                  |  
|------------------------|------------|----------------------------------------------|  
| `bayesian_mcmc.py`     | Python Script | Bayesian MCMC sampling (posterior for prices, demand shocks, cost shocks) |  
| `blp.py`               | Python Script | Core BLP logic (GMM objective computation, optimization) |  
| `data_generation.py`   | Python Script | Synthetic data generator (markets/individuals/products tensors) |  
| `main_comparison.py`    | Python Script | Orchestrates pipeline: data generation → estimation → comparison |  
| `test.py`              | Python Script | Unit/integration test suite|  

