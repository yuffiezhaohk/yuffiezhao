# DeepHalo Model Replication

This repository contains a replication of the **DeepHalo** model introduced by Zhang et al. (2025). The system is built using **TensorFlow** and **Choice-Learn**.  

## 📂 Repository Structure  

### 1. Code Directory (`/code`)  
Contains core implementation files:  

| Filename          | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `main.py`         | Orchestrates the full pipeline (data generation → visualization).          |
| `config.py`       | Manages hyperparameters (universe size, budgets, network depths).           |
| `data_loader.py`  | Generates synthetic data and wraps it in `ChoiceDataset`.                  |
| `model.py`        | Implements DeepHalo architecture with:<br>- Quadratic Activations<br>- ResNet topology<br>- Availability Masking. |
| `experiment.py`   | Handles training loops and evaluates Distribution-Matching RMSE.           |
| `utils.py`        | Provides architectural constraint solver and random seeding. |
| `visualization.py`| Generates comparative performance and convergence plots.                   |
| `export.py`       | Handles timestamped saving of models, metadata, and JSON results.          |
| `test.py`         | Contains comprehensive unit and integration test suite.                    |


### 2. Root Directory  
Contains documentation and reports:  

| Filename              | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `testing_plan.md`     | Validation strategy for mathematical, data, and model logic.                |
| `testing_report.md`   | Log of successful validation tests for the modular system.                  |
| `report.pdf`          | Formal summary and analysis of reproduction results.                       |